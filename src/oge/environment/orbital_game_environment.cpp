//
// Created by baoyicui on 2/22/26.
//

#include "orbital_game_environment.h"

// #include <csignal>


oge::OrbitalGameEnvironment::OrbitalGameEnvironment() :
    dv_max_per_step_e(0.01),
    dv_max_per_step_p(0.01),
    timestep(60),
    terminal_time(7200),
    capture_distance(30)
{
}

bool oge::OrbitalGameEnvironment::isTerminal() const
{
    for (int i = 1; i < agents_states.size(); ++i)
    {
        if ((agents_states[0].r_j2000 - agents_states[i].r_j2000).norm() < capture_distance)
        {
            return true;
        }
    }
    return false;
}

bool oge::OrbitalGameEnvironment::isTruncated() const
{
    for (int i = 1; i < agents_states.size(); ++i)
    {
        // 检查pursuers的燃料是否耗尽
    }

    return current_time >= terminal_time;
}

void oge::OrbitalGameEnvironment::reset()
{
    // TODO: 在这里初始化状态和燃料
}

void oge::OrbitalGameEnvironment::processDynamics(std::vector<Eigen::Vector3d>& actions)
{
    if (actions.size() != agents_states.size())
    {
        throw std::invalid_argument("actions.size() != agents_states.size()");
    }

    for (int i = 0; i < actions.size(); ++i)
    {
        // constraints on dv
        Eigen::Vector3d dv_modified;
        if (actions[i].norm() > std::min(dv_max_per_step, agents_states[i].dv_remain))
        {
            dv_modified = std::min(dv_max_per_step, agents_states[i].dv_remain) * actions[i].normalized();
        }
        else
        {
            dv_modified = actions[i];
        }
        // update agent's velocity in J2000
        agents_states[i].v_j2000 += dv_modified;

        // propagation
        Eigen::Vector3d r_j2000_new, v_j2000_new;
        rv_from_r0v0(agents_states[i].r_j2000, agents_states[i].v_j2000, timestep, r_j2000_new, v_j2000_new);
        agents_states[i].r_j2000 = r_j2000_new;
        agents_states[i].v_j2000 = v_j2000_new;
    }
    current_time += timestep;
}

void oge::OrbitalGameEnvironment::act(
    std::vector<Eigen::Vector3d>& agents_actions,
    std::vector<double>& agents_rewards)
{
    if (agents_actions.size() != agents_states.size())
    {
        throw std::invalid_argument("agents_actions.size() != agents_states.size()");
    }
    processDynamics(agents_actions);

    // TODO: get observations

    // TODO: get rewards

    // TODO: get truncations

    // TODO: get terminations
}
