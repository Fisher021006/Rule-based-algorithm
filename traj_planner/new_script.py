import numpy as np
import osqp
from scipy import sparse
from scipy.sparse import eye, vstack
from matplotlib import pyplot as plt


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bicycle_dynamics(x, u, l=2.5, dt=0.1, max_delta=np.pi / 4):
    x_pos, y_pos, v, theta = x
    a, delta = u
    delta = np.clip(delta, -max_delta, max_delta)
    x_next = x_pos + v * np.cos(theta) * dt
    y_next = y_pos + v * np.sin(theta) * dt
    v_next = v + a * dt
    theta_next = theta + (v / l) * np.tan(delta) * dt
    return np.array([x_next, y_next, v_next, theta_next])


def bicycle_jacobian(x, u, l=2.5, dt=0.1):
    _, _, v, theta = x
    _, delta = u

    A = np.eye(4)
    B = np.zeros((4, 2))

    A[0, 2] = np.cos(theta) * dt
    A[0, 3] = -v * np.sin(theta) * dt

    A[1, 2] = np.sin(theta) * dt
    A[1, 3] = v * np.cos(theta) * dt

    A[3, 2] = (1.0 / l) * np.tan(delta) * dt

    B[2, 0] = dt
    B[3, 1] = (v / l) * (1.0 / np.cos(delta) ** 2) * dt

    return A, B


class BicycleSCP:
    def __init__(self, config, dt=0.1, verbose=False):
        self.dt = dt
        self.l = config["mpc"]["l"]
        self.max_iter = config["mpc"]["max_iter"]

        self.w_xy = config["mpc"]["w_xy"]
        self.w_a = config["mpc"]['w_a']
        self.w_delta = config["mpc"]['w_delta']

        self.w_da = config["mpc"]["w_da"]
        self.w_ddelta = config["mpc"]["w_ddelta"]

        self.w_xy_end = config["mpc"]["w_xy_end"]
        self.w_v_end = config["mpc"]["w_v_end"]
        self.w_a_end = config["mpc"]["w_a_end"]
        self.w_theta_end = config["mpc"]["w_theta_end"]

        self.acc_min = config["constrain"]["min_acc"]
        self.acc_max = config["constrain"]["max_acc"]
        self.vel_min = config["constrain"]["min_v"]
        self.vel_max = config["constrain"]["max_v"]
        self.delta_min = -np.arctan(config["constrain"]["max_kappa"] * self.l)
        self.delta_max = np.arctan(config["constrain"]["max_kappa"] * self.l)

        self.nx = 4
        self.nu = 2
        self.verbose = verbose

    # -----------------------------------------------------
    def rollout(self, x0, U, N):
        X = np.zeros((N + 1, self.nx))
        X[0] = x0[[0, 1, 2, 4]]
        for k in range(N):
            X[k + 1] = bicycle_dynamics(X[k], U[k], self.l, self.dt, self.delta_max)
        return X

    # -----------------------------------------------------
    def solve(self, x0, xT, ref_xy):
        N = len(ref_xy) + 1
        nx, nu = self.nx, self.nu

        ref = np.vstack([[x0[0], x0[1]], ref_xy, [xT[0], xT[1]]])

        U = np.zeros((N, nu))
        X = self.rollout(x0, U, N)

        for it in range(self.max_iter):
            A_list, B_list = [], []
            for k in range(N):
                A, B = bicycle_jacobian(X[k], U[k], self.l, self.dt)
                A_list.append(A)
                B_list.append(B)

            z, ok = self._build_and_solve_qp(
                N, X, U, A_list, B_list, x0, xT, ref
            )

            if not ok:
                return None, None, False

            dX = z[: (N + 1) * nx].reshape((N + 1, nx))
            dU = z[(N + 1) * nx:].reshape((N, nu))

            X += dX
            U += dU

            err = np.max(np.abs(dX)) + np.max(np.abs(dU))
            if self.verbose:
                print(f"[SCP] iter {it}, update {err:.3e}")

            if err < 1e-2:
                break

        return X, U, True

    # -----------------------------------------------------
    def _build_and_solve_qp(self, N, X, U, A_list, B_list, x0, xT, ref):
        nx, nu = self.nx, self.nu
        n_var = (N + 1) * nx + N * nu

        P_rows, P_cols, P_data = [], [], []
        q = np.zeros(n_var)

        # 中间点的追踪损失
        for k in range(1, N):
            base = k * nx
            for i in [0, 1]:
                P_rows.append(base + i)
                P_cols.append(base + i)
                P_data.append(self.w_xy)
                q[base + i] = self.w_xy * (X[k, i] - ref[k, i])

        # 终端损失
        tb = N * nx

        # 终端的追踪损失
        for i in [0, 1]:
            P_rows.append(tb + i)
            P_cols.append(tb + i)
            P_data.append(self.w_xy_end)
            q[tb + i] = self.w_xy_end * (X[N, i] - ref[N, i])

        # 终端的速度损失
        P_rows.append(tb + 2)
        P_cols.append(tb + 2)
        P_data.append(self.w_v_end)
        q[tb + 2] = self.w_v_end * (X[N, 2] - xT[2])

        # 终端的角度损失
        P_rows.append(tb + 3)
        P_cols.append(tb + 3)
        P_data.append(self.w_theta_end)
        q[tb + 3] = self.w_theta_end * wrap_angle(X[N, 3] - xT[4])

        # 终端加速度损失
        ia = (N + 1) * nx + (N - 1) * nu
        P_rows.append(ia)
        P_cols.append(ia)
        P_data.append(self.w_a_end)
        q[ia] = self.w_a_end * U[N - 1, 0]

        # 控制量a与δ的大小损失
        for k in range(N):
            base_u = (N + 1) * nx + k * nu
            idx_a = base_u + 0
            P_rows.append(idx_a)
            P_cols.append(idx_a)
            P_data.append(self.w_a)
            q[idx_a] += self.w_a * U[k, 0]
            idx_d = base_u + 1
            P_rows.append(idx_d)
            P_cols.append(idx_d)
            P_data.append(self.w_delta)
            q[idx_d] += self.w_delta * U[k, 1]

        # 控制量变化量Δa与Δδ的大小损失
        for k in range(1, N):
            uk = (N + 1) * nx + k * nu
            ukm1 = (N + 1) * nx + (k - 1) * nu
            for i, w in zip([0, 1], [self.w_da, self.w_ddelta]):
                P_rows += [uk + i, ukm1 + i, uk + i, ukm1 + i]
                P_cols += [uk + i, ukm1 + i, ukm1 + i, uk + i]
                P_data += [w, w, -w, -w]

        # 问题整合
        P = sparse.csc_matrix((P_data, (P_rows, P_cols)), shape=(n_var, n_var))

        # 运动学约束
        rows, cols, data = [], [], []
        b = np.zeros((N + 1) * nx)

        # x0
        for i in range(nx):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
        b[:nx] = x0[[0, 1, 2, 4]] - X[0]

        r = nx
        for k in range(N):
            fk = bicycle_dynamics(X[k], U[k], self.l, self.dt, self.delta_max)
            b[r: r + nx] = -(X[k + 1] - fk)

            for i in range(nx):
                rows.append(r + i)
                cols.append((k + 1) * nx + i)
                data.append(1.0)
                for j in range(nx):
                    rows.append(r + i)
                    cols.append(k * nx + j)
                    data.append(-A_list[k][i, j])
                for j in range(nu):
                    rows.append(r + i)
                    cols.append((N + 1) * nx + k * nu + j)
                    data.append(-B_list[k][i, j])
            r += nx

        Aeq = sparse.csc_matrix((data, (rows, cols)), shape=((N + 1) * nx, n_var))
        leq = b.copy()
        ueq = b.copy()

        # --------- box constraints ----------
        lb = -np.inf * np.ones(n_var)
        ub = np.inf * np.ones(n_var)

        for k in range(N + 1):
            lb[k * nx + 2] = self.vel_min - X[k, 2]
            ub[k * nx + 2] = self.vel_max - X[k, 2]

        for k in range(N):
            b0 = (N + 1) * nx + k * nu
            lb[b0] = self.acc_min - U[k, 0]
            ub[b0] = self.acc_max - U[k, 0]
            lb[b0 + 1] = self.delta_min - U[k, 1]
            ub[b0 + 1] = self.delta_max - U[k, 1]

        A = vstack([Aeq, eye(n_var)], format="csc")
        l = np.hstack([leq, lb])
        u = np.hstack([ueq, ub])

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False)
        res = prob.solve()

        if res.info.status_val == osqp.constant("OSQP_SOLVED"):
            return res.x, True
        return None, False


# =========================================================
# Example
# =========================================================
# if __name__ == "__main__":
#     start = np.array([1.0, 1.0, 5.0, 0.0, np.pi / 2])  # x, y, v, a, theta
#     goal = np.array([21.0, 5.0, 5.0, 0.0, 0.0])
#
#     N = 40
#     ref = np.array([(0.5 * i + 1, 0.1 * i + 1) for i in range(1, N)])
#
#     config = {
#         "constrain": {
#             "min_v": 0.0,
#             "max_v": 120 / 3.6,
#             "min_acc": -6.0,
#             "max_acc": 3.0,
#             "max_kappa": 0.3,
#         },
#         "mpc": {
#             "l": 2.5,
#             "max_iter": 1000,
#             "w_xy": 0.025,
#             "w_a": 0.1,
#             "w_delta": 0.5,
#             "w_da": 1,
#             "w_ddelta": 1,
#             "w_xy_end": 1.0,
#             "w_v_end": 2.0,
#             "w_theta_end": 3.0,
#             "w_a_end": 1.0,
#         },
#     }
#
#     solver = BicycleSCP(config, verbose=False)
#     import time
#
#     st = time.perf_counter()
#     X, U, ok = solver.solve(start, goal, ref[:, 0:2])
#     print((time.perf_counter() - st) * 1000)
#     print("success:", ok)
#     if ok:
#         print("final state:", X[-1])
#
#     plt.plot(X[:, 0], X[:, 1], "b")
#     plt.axis('equal')
#     plt.show()
#     plt.plot(X[:, 2], "b")
#     plt.show()
#     plt.plot(X[:, 3], "b")
#     plt.show()
#     plt.plot(U[:, 0])
#     plt.show()
#     plt.plot(U[:, 1])
#     plt.show()
