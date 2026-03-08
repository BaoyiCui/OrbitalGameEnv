//
// Created by baoyicui on 2/22/26.
//

#include "orbital_game_environment.h"

namespace oge
{
    OrbitalGameEnvironment::OrbitalGameEnvironment(
        OGESettings& settings_
    ) :
        settings(settings_),
        dv_max_per_step_p(settings_.dv_max_per_step_p),
        dv_max_per_step_e(settings_.dv_max_per_step_e),
        capture_distance(settings_.capture_distance),
        timestep(settings_.timestep),
        terminal_time(settings_.terminal_time),
        num_pursuers(settings_.num_pursuers),
        num_evaders(settings_.num_evaders),
        num_agents(settings_.num_pursuers + settings_.num_evaders),
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

    void OrbitalGameEnvironment::getRewards(const std::vector<Eigen::Vector3d>& agent_actions,
                                            std::vector<double>& rewards) const
    {
        rewards.resize(num_agents);

        std::vector<double> dists_to_evader;
        dists_to_evader.reserve(num_pursuers);
    }


    void OrbitalGameEnvironment::reset()
    {
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

    double OrbitalGameEnvironment::getFormationReward() const
    {
        // Formation reward only makes sense when there are multiple pursuers.
        if (num_pursuers < 2)
            return 0.0;

        // Accumulate the unit direction vectors from the evader (index 0) to each pursuer.
        // If pursuers surround the evader uniformly, their unit vectors cancel out and
        // sum_directions approaches zero — which is the ideal formation.
        Eigen::Vector3d sum_directions = Eigen::Vector3d::Zero();
        for (int p = num_evaders; p < num_agents; ++p)
        {
            double r_diff_j2000_norm = (agents_states[p].r_j2000 - agents_states[0].r_j2000).norm();
            // Skip pursuers that coincide with the evader to avoid division by zero.
            if (almost_equal(r_diff_j2000_norm, 0.0))
                continue;
            sum_directions += (agents_states[p].r_j2000 - agents_states[0].r_j2000) / r_diff_j2000_norm;
        }

        // reward = weight / (1 + ||sum_directions||)
        // The norm of sum_directions is 0 for perfect encirclement and up to num_pursuers
        // when all pursuers are on the same side. Dividing 1 by (1 + norm) maps this to (0, 1].
        const double reward_formation = settings.reward_formation_weight * (1.0 / (1.0 + sum_directions.norm()));

        return reward_formation;
    }

    double OrbitalGameEnvironment::getDistanceReward(int p) const
    {
        // TODO
        double distance = (agents_states[p].r_j2000 - agents_states[0].r_j2000).norm();

        Eigen::Matrix<double, 6, 1> coe_p, coe_e;
        rv2coe(agents_states[p].r_j2000, agents_states[p].v_j2000, coe_p);
        rv2coe(agents_states[0].r_j2000, agents_states[0].v_j2000, coe_e);

        double TA_delta = std::fmod((coe_p - coe_e)(5) + M_PI, 2.0 * M_PI) - M_PI;
        double sma_diff_ratio = (coe_p - coe_e)(0) / coe_e(0);

        // Far field
        double drift_product = TA_delta * sma_diff_ratio;
        double reward_far, reward_near;
        if (drift_product > 0.0)
        {
            reward_far = -1.0 - std::abs(sma_diff_ratio) * 2000.0;
        }
        else
        {
            // python code:
            // r_drift = np.clip(np.abs(sma_diff_ratio) * 1000.0, 0.0, 2.0)
            // r_angle = (np.pi - np.abs(delta_theta)) / np.pi
            // R_Far = 1.0 * r_drift + 0.5 * r_angle
            double r_drift = std::clamp(std::abs(sma_diff_ratio) * 1000.0, 0.0, 2.0);
            double r_angle = (M_PI - std::abs(TA_delta)) / M_PI;
            reward_far = 1.0 * r_drift + 0.5 * r_angle;
        }

        // TODO: Near field
        double dist_normalized = distance / settings.capture_distance;
        if (dist_normalized <= 1.0)
        {
        }
        else if (dist_normalized <= 2.0)
        {
        }
        else
        {
        }
        double reward_energy = ;
        reward_near =;

        double alpha = std::abs(sma_diff_ratio) * 2000.0;
        double total_reward = alpha * reward_far + (1.0 - alpha) * reward_near;


        return settings.reward_phase_dist_weight * total_reward;
    }

    double OrbitalGameEnvironment::getCaptureReward(int p) const
    {
        bool captured_team = false;
        for (int i = num_evaders; i < num_agents; ++i)
        {
            if ((agents_states[i].r_j2000 - agents_states[0].r_j2000).norm() < settings.capture_distance)
            {
                captured_team = true;
                break;
            }
        }

        if (captured_team)
        {
            if ((agents_states[p].r_j2000 - agents_states[0].r_j2000).norm() < settings.capture_distance)
            {
                return settings.reward_capture_weight; // capture bonus
            }

            return 0.5 * settings.reward_capture_weight; // assistant capture bonus
        }

        return 0.0; // no capture
    }

    double OrbitalGameEnvironment::getFuelReward(int p, const std::vector<Eigen::Vector3d>& actions) const
    {
        return settings.reward_fuel_weight * actions[p - num_evaders].norm();;
    }

    double OrbitalGameEnvironment::getTimeReward() const
    {
        return settings.reward_time_weight;
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
        getRewards(agents_actions, agents_rewards);

        // TODO: get truncations

        // TODO: get terminations
    }
}
