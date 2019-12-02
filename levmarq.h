//
// Created by spc on 19-11-30.
//

#ifndef USELEVENBERGMARQUARDT_LEVMARQ_H
#define USELEVENBERGMARQUARDT_LEVMARQ_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;

class LeverbergMarquardt
{
public:
    LeverbergMarquardt(Eigen::Vector3d abc): _abc(abc)
    {
        _epsilon1 = 1e-6;
        _epsilon2 = 1e-6;
        _max_iter = 50;
        _is_out = true;
    }

    ~LeverbergMarquardt() {}

    // 设置相关参数
    void setParameters(double epsilon1, double epsilon2, int max_iter, bool is_out)
    {
        _epsilon1 = epsilon1;
        _epsilon2 = epsilon2;
        _max_iter = max_iter;
        _is_out = is_out;
    }

    // 获取结果
    Eigen::Vector3d getResult()
    {
        Eigen::Vector3d res;
        res(0) = _abc(0);
        res(1) = _abc(1);
        res(2) = _abc(2);
        return res;
    }

    // 加入测量值
    void addMeasurement(const double& x, const double& y)
    {
        _meas.push_back(Eigen::Vector2d(x, y));
    }

    // 计算J, f(x), H, g
    void compute(Eigen::MatrixXd& fx, Eigen::MatrixXd& J, Eigen::Matrix3d& H,
                 Eigen::Vector3d& g, const Eigen::Vector3d& abc, bool is_calc_J);

    // 计算目标函数值
    // double calc_object(const Eigen::Vector3d& abc);

    // 计算二阶近似的差值( m(0) - m(p) )
    double calc_m(const Eigen::Vector3d& p, const Eigen::MatrixXd& fx, const Eigen::MatrixXd& J)
    {
        Eigen::MatrixXd m = -p.transpose() * J.transpose() * fx - 0.5 * p.transpose() * J.transpose() * J * p;
        return m(0, 0);
    }

    // 整个优化过程
    void Solve();

protected:

    Eigen::Vector3d _abc;             // 要优化的三个参数
    vector<Eigen::Vector2d> _meas;
    double _epsilon1, _epsilon2;
    int _max_iter;                  // 最大迭代次数
    bool _is_out;                   // 是否输出优化信息
};

#endif //USELEVENBERGMARQUARDT_LEVMARQ_H
