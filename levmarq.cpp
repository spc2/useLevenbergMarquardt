#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "levmarq.h"

using namespace std;

void LeverbergMarquardt::compute(Eigen::MatrixXd &fx, Eigen::MatrixXd &J, Eigen::Matrix3d &H,
                                 Eigen::Vector3d &g, const Eigen::Vector3d& abc, bool is_calc_J)
{
    fx.resize(_meas.size(), 1);
    for (size_t i = 0; i < _meas.size(); i++)
    {
        const Eigen::Vector2d& temp = _meas.at(i);
        const double& x = temp(0);
        const double& y = temp(1);
        fx(i, 0) = y - exp(abc(0)*x*x + abc(1)*x + abc(2));
        if (is_calc_J)      // 是否计算Jacobian
        {
            J.resize(_meas.size(), 3);
            J(i, 0) = -x * x * exp(abc(0)*x*x + abc(1)*x + abc(2));
            J(i, 1) = -x * exp(abc(0)*x*x + abc(1)*x + abc(2));
            J(i, 2) = -exp(abc(0)*x*x + abc(1)*x + abc(2));
        }
    }

    if (is_calc_J)
    {
        H = J.transpose() * J;
        g = -J.transpose() * fx;
    }
}

/*double LeverbergMarquardt::calc_object(const Eigen::Vector3d& abc)
{
    Eigen::MatrixXd fx;
    fx.resize(_meas.size(), 1);
    for (size_t i = 0; i < _meas.size(); i++)
    {
        const Eigen::Vector2d& temp = _meas.at(i);
        const double& x = temp(0);
        const double& y = temp(1);
        fx(i, 0) = y - exp(abc(0)*x*x + abc(1)*x + abc(2));
    }
    Eigen::MatrixXd val = 0.5 * fx.transpose() * fx;
    return val(0, 0);
} */

void LeverbergMarquardt::Solve()
{
    // 初始化
    int iter = 0;       // 迭代次数
    double nu = 2.0;    // 初始信赖域大小
    const double rho_l = 0.25, rho_h = 0.75;
    double lambda = 1, lc = 0.75;
    Eigen::MatrixXd J, fx, fxp;
    J.resize(_meas.size(), 3);
    fx.resize(_meas.size(), 1);
    fxp.resize(_meas.size(), 1);
    Eigen::Matrix3d H, Hp;
    Eigen::Vector3d g, p, abc;

    // 记录时间
    double sum = 0;
    chrono::steady_clock::time_point t1;
    chrono::steady_clock::time_point t2;

    compute(fx, J, H, g, _abc, true);
    double S = 0.5 * fx.squaredNorm();
    Eigen::Vector3d D = H.diagonal();   // 取出H的对角线元素

    bool proceed = true;    // 是否继续处理
    while (proceed && iter < _max_iter)
    {
        t1 = chrono::steady_clock::now();
        Hp = H;
        for (int i = 0; i < H.rows(); i++)
        {
            Hp(i, i) += lambda * D(i);
        }

        p = Hp.ldlt().solve(g); // 求解增量方程: (H + lambda*D^TD)p = g
        abc = _abc + p;
        compute(fxp, J, H, g, abc, false);

        double Sd = 0.5 * fxp.squaredNorm();
        double dS = calc_m(p, fx, J);
        double rho = (S - Sd) / (dS > DBL_EPSILON ? dS: 1);

        if (rho > rho_h)        // 二阶近似的效果好, 缩小lambda
        {
            lambda *= 0.5;
            if (lambda < lc)
                lambda = 0;
        }
        else if (rho < rho_l)  // 二阶近似效果差, 扩大lambda, 增大信赖域半径
        {
            double t = p.dot(g);
            nu = (Sd - S) / (fabs(t) > DBL_EPSILON ? t : 1) + 2;
            nu = min(max(nu, 2.), 10.);
            if (lambda == 0)
            {
                Hp = H.inverse();
                auto maxval = DBL_EPSILON;
                for (int i = 0; i < H.rows(); i++)
                {
                    maxval = max(maxval, abs(Hp(i, i)));
                }
                lambda = lc = 1./maxval;
                nu *= 0.5;
            }
            lambda *= nu;
        }

        if (S > Sd)
        {
            S = Sd;
            _abc = abc;
            compute(fx, J, H, g, _abc, true);
        }

        iter++;
        proceed = p.lpNorm<Eigen::Infinity>() >= _epsilon1 && fx.lpNorm<Eigen::Infinity>() >= _epsilon2;

        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

        if (_is_out)
        {
            cout << "Iter: " << left << setw(3) << iter << "Result: " << left << setw(10) << _abc(0) << " " <<
            left << setw(10) << _abc(1) << " " << left << setw(10) << _abc(2) << " Step: " << left << setw(14) <<
            p.norm() << "Cost: " << left << setw(14) << 0.5 * fx.squaredNorm() << "Time: " << left << setw(14) <<
            time_used.count() << "Total_time: " << left << setw(14) << (sum += time_used.count()) << endl;
        }
    }

    if (!proceed)
        cout << endl << "Converged!" << endl;
    else
        cout << endl << "Diverged" << endl;
}


int main()
{
    const Eigen::Vector3d actual_abc(1.0, 2.0, 1.0);  // 方程的实际参数
    Eigen::Vector3d abc(0, 0, 0);                     // 初值
    ofstream outfile;                                 // 保存数据
    outfile.open("data.txt", ios::trunc);

    // 构造问题
    LeverbergMarquardt lm(abc);
    lm.setParameters(FLT_EPSILON, FLT_EPSILON, 100, true);

    // 产生带噪声的数据
    const size_t N = 100;
    cv::RNG rng;
    double w_sigma = 1.0;
    for (size_t i = 0; i < N; i++)
    {
        double x = i / 100.0;
        double y = exp(actual_abc(0)*x*x + actual_abc(1)*x + actual_abc(2)) + rng.gaussian(w_sigma);

        outfile << x << " " << y << endl;

        // 加入到测量中
        lm.addMeasurement(x, y);
    }

    // 求解
    lm.Solve();

    abc = lm.getResult();
    outfile << actual_abc(0) << " " << actual_abc(1) << " " << actual_abc(2) << endl;
    outfile << abc(0) << " " << abc(1) << " " << abc(2) << endl;
    outfile.close();

    return 0;
}