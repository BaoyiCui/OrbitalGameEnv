//
// Created by baoyicui on 2/22/26.
//

#include "orbital_game_environment.h"

namespace oge
{
    OrbitalGameEnvironment::OrbitalGameEnvironment(
        OGESettings& settings
    ) :
        num_pursuers(settings.num_pursuers),
        num_evaders(settings.num_evaders),
        num_agents(settings.num_pursuers + settings.num_evaders),
        dv_max_per_step_e(settings.dv_max_per_step_e),
        dv_max_per_step_p(settings.dv_max_per_step_p),
        timestep(settings.timestep),
        terminal_time(settings.terminal_time),
        capture_distance(settings.capture_distance)
    {
        agent_ids.reserve(num_agents);
        agents_states.reserve(num_agents);
        // initialize agent_ids
        for (int i = 0; i < num_evaders; ++i)
        {
            agent_ids.push_back("e_" + std::to_string(i));
        }
        for (int i = 0; i < num_pursuers; ++i)
        {
            agent_ids.push_back("p_" + std::to_string(i));
        }
        // initialize agents_
    }

    bool OrbitalGameEnvironment::isTerminal() const
    {
        const bool all_captured = std::all_of(
            agents_states.begin(),
            agents_states.begin() + num_evaders,
            [](const SatState& s) { return !s.is_alive; }
        );
        if (all_captured) return true;

        const bool fuel_exceeded = std::all_of(
            agents_states.begin() + num_evaders,
            agents_states.end(),
            [](const SatState& s) { return !s.is_alive; }
        );

        if (fuel_exceeded) return true;

        return false;
    }

    bool OrbitalGameEnvironment::isTruncated() const
    {
        return current_time >= terminal_time;
    }

    void OrbitalGameEnvironment::getObservations(std::vector<Eigen::Matrix<double, 18, 1>>& observations)
    {
        // calculate evader's observation

        // calculate pursuer's observation
        for (int i = num_evaders; i < num_agents; ++i)
        {
        }
    }


    void OrbitalGameEnvironment::reset()
    {
        // TODO: 在这里初始化状态和燃料
    }

    void OrbitalGameEnvironment::processDynamics(std::vector<Eigen::Vector3d>& actions)
    {
        if (actions.size() != num_agents)
        {
            throw std::invalid_argument("actions.size() != num_agents");
        }

        for (int i = 0; i < num_agents; ++i)
        {
            if (!agents_states[i].is_alive)
            {
                continue;
            }

            // constraints on dv
            Eigen::Vector3d dv_modified;
            double dv_max_per_step = i == 0 ? dv_max_per_step_e : dv_max_per_step_p;
            if (agents_states[i].dv_remain <= 0.0)
            {
                throw; // TODO 这里如果抛出异常说明act中的顺序有问题
            }
            if (almost_equal(actions[i].norm(), 0.0))
            {
                if (actions[i].norm() > std::min(dv_max_per_step, agents_states[i].dv_remain))
                {
                    dv_modified = std::min(dv_max_per_step, agents_states[i].dv_remain) * actions[i].normalized();
                }
                else
                {
                    dv_modified = actions[i];
                }
            }
            else
            {
                dv_modified = Eigen::Vector3d::Zero();
            }
            // update agent's velocity in J2000
            agents_states[i].v_j2000 += dv_modified;
            // update agent's fuel
            agents_states[i].dv_remain -= dv_modified.norm();

            // propagation
            Eigen::Vector3d r_j2000_new, v_j2000_new;
            rv_from_r0v0(agents_states[i].r_j2000, agents_states[i].v_j2000, timestep, r_j2000_new, v_j2000_new);
            agents_states[i].r_j2000 = r_j2000_new;
            agents_states[i].v_j2000 = v_j2000_new;
        }
        current_time += timestep;
    }

    void OrbitalGameEnvironment::checkAlive()
    {
        for (int e = 0; e < num_evaders; ++e)
        {
            for (int p = num_evaders; p < num_agents; ++p)
            {
                if ((agents_states[e].r_j2000 - agents_states[p].r_j2000).norm() < capture_distance)
                {
                    agents_states[e].is_alive = false;
                    break;
                }
            }
        }
        for (int p = num_evaders; p < num_agents; ++p)
        {
            // 检查pursuers的燃料是否耗尽
            if (almost_equal(agents_states[p].dv_remain, 0.0))
            {
                agents_states[p].is_alive = false;
            }
        }
    }

    void OrbitalGameEnvironment::act(
        std::vector<Eigen::Vector3d>& agents_actions,
        std::vector<double>& agents_rewards)
    {
        if (agents_actions.size() != num_agents)
        {
            throw std::invalid_argument("agents_actions.size() != num_agents");
        }
        processDynamics(agents_actions);
        checkAlive();

        // TODO: get observations

        // TODO: get rewards

        // TODO: get truncations

        // TODO: get terminations
    }
}
