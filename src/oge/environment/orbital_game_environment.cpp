//
// Created by baoyicui on 2/22/26.
//

#include "orbital_game_environment.h"

namespace oge
{
    OrbitalGameEnvironment::OrbitalGameEnvironment(
        OGESettings& settings
    ) :
        dv_max_per_step_p(settings.dv_max_per_step_p),
        dv_max_per_step_e(settings.dv_max_per_step_e),
        capture_distance(settings.capture_distance),
        timestep(settings.timestep),
        terminal_time(settings.terminal_time),
        num_pursuers(settings.num_pursuers),
        num_evaders(settings.num_evaders),
        num_agents(settings.num_pursuers + settings.num_evaders),
        current_time(0.0)
    {
        settings.validate();
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

    int OrbitalGameEnvironment::getObsSize(int agent_idx) const
    {
        // For pursuer
        //      The first 6 elements are RV in J2000 frame.
        //      The last 3 elements are target position in this pursuer's LVLH frame.
        //      The remaining elements are other pursuers' positions in this pursuer's LVLH frame.
        // For evader
        //      The first 6 elements are RV in J2000 frame.
        //      The other elements are pursuers' positions in this evader's LVLH frame.
        // This observation's structure only supports OGE with a single evader.
        return 3 * (num_agents + 1);
    }

    void OrbitalGameEnvironment::getObservations(std::vector<Eigen::VectorXd>& observations)
    {
        observations.resize(num_agents);
        for (int e = 0; e < num_evaders; ++e)
        {
            observations[e].resize(getObsSize(e));
            observations[e].segment<3>(0) = agents_states[e].r_j2000;
            observations[e].segment<3>(3) = agents_states[e].v_j2000;
            for (int p = num_evaders; p < num_agents; ++p)
            {
                Eigen::Vector3d r_p_lvlh;
                Eigen::Vector3d v_p_lvlh;
                RV_J20002LVLH(
                    agents_states[e].r_j2000, agents_states[e].v_j2000,
                    agents_states[p].r_j2000, agents_states[p].v_j2000,
                    r_p_lvlh, v_p_lvlh
                );
                observations[e].segment<3>(6 + 3 * (p - num_evaders)) = r_p_lvlh;
            }
        }

        for (int p = num_evaders; p < num_agents; ++p)
        {
            observations[p].resize(getObsSize(p));
            observations[p].segment<3>(0) = agents_states[p].r_j2000;
            observations[p].segment<3>(3) = agents_states[p].v_j2000;

            // other pursuers' positions in this pursuer's LVLH frame
            int offset = 6;
            for (int other_p = num_evaders; other_p < num_agents; ++other_p)
            {
                if (other_p == p)
                    continue;

                Eigen::Vector3d r_other_p_lvlh, v_other_p_lvlh;
                RV_J20002LVLH(
                    agents_states[p].r_j2000, agents_states[p].v_j2000,
                    agents_states[other_p].r_j2000, agents_states[other_p].v_j2000,
                    r_other_p_lvlh, v_other_p_lvlh
                );
                observations[p].segment<3>(offset) = r_other_p_lvlh;
                offset += 3;
            }

            // evader (target) position in this pursuer's LVLH frame — last 3 elements
            Eigen::Vector3d r_e_lvlh, v_e_lvlh;
            RV_J20002LVLH(
                agents_states[p].r_j2000, agents_states[p].v_j2000,
                agents_states[0].r_j2000, agents_states[0].v_j2000,
                r_e_lvlh, v_e_lvlh
            );
            observations[p].segment<3>(getObsSize(p) - 3) = r_e_lvlh;
        }
    }

    void OrbitalGameEnvironment::getRewards(std::vector<double>& rewards)
    {
        // calculate evader's reward
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
            double dv_max_per_step = (i < num_evaders) ? dv_max_per_step_e : dv_max_per_step_p;
            if (agents_states[i].dv_remain <= 0.0)
            {
                throw std::runtime_error("Agent " + std::to_string(i) + " has no remaining dv but is still alive");
            }
            if (!almost_equal(actions[i].norm(), 0.0))
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
