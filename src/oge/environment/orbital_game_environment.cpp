//
// Created by baoyicui on 2/22/26.
//

#include "orbital_game_environment.h"

oge::OrbitalGameEnvironment::OrbitalGameEnvironment() :
    dv_max_per_step(0.01), timestep(60), terminal_time(7200)
{
}

bool oge::OrbitalGameEnvironment::isTerminal() const
{
    if ()

}

bool oge::OrbitalGameEnvironment::isTruncated() const
{
    if (current_time >= terminal_time)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void oge::OrbitalGameEnvironment::reset()
{
}

void oge::OrbitalGameEnvironment::processDynamics(std::vector<Eigen::Vector3d> actions)
{
    for (int i = 0; i < actions.size(); ++i)
    {
        // constraints on dv
        Eigen::Vector3d dv_modified;
        if (actions[i].norm() > std::min(dv_max_per_step, agent_states[i].dv_remain))
        {
            dv_modified = std::min(dv_max_per_step, agent_states[i].dv_remain) * actions[i].normalized();
        }
        else
        {
            dv_modified = actions[i];
        }
        // update agent's velocity in J2000
        agent_states[i].v_j2000 += dv_modified;

        // propagation
        Eigen::Vector3d r_j2000_new, v_j2000_new;
        rv_from_r0v0(agent_states[i].r_j2000, agent_states[i].v_j2000, timestep, r_j2000_new, v_j2000_new);
        agent_states[i].r_j2000 = r_j2000_new;
        agent_states[i].v_j2000 = v_j2000_new;
    }
    current_time += timestep;
}

void oge::OrbitalGameEnvironment::act(
    std::vector<Eigen::Vector3d> pursuers_actions,
    Eigen::Vector3d evader_actions,
    std::vector<double>& pursuers_rewards)
{
}
