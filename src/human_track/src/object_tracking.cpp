#include <object_tracking.hpp>

//using namespace std;
using std::deque;
using std::map;
using NEWMAT::Matrix;
using NEWMAT::IdentityMatrix;

using namespace Header;

namespace ObjectTracking {
    float max_mah_dist;
    float max_euc_dist;

    int init_id;

    float new_object_score;
    float predict_score;
    float update_score;
}

void ObjectTracking::loadCfg(string cfg) {

    max_mah_dist = 1.8;//1.8
   //max_mah_dist = 3;
    max_euc_dist = 2;//2
    //max_euc_dist = 3.2;

    init_id = 0;

    new_object_score = 5;
    predict_score = 3;
    update_score = 10;
}

void ObjectTracking::loadCfg(detectorParameters params) {
    max_mah_dist = params.max_mah_dist;
    max_euc_dist = params.max_euc_dist;

    init_id = params.init_id;

    new_object_score = params.new_object_score;
    predict_score = params.predict_score;
    update_score = params.update_score;
}

void ObjectTracking::eliminate(deque<HumanCluster> &humans) {
    for (int i = 0; i < (int) humans.size(); i++) {
        //if(i==1)
        if (humans[i].score < 0.0 || curTimestamp - humans[i].preTimestamp > 3) {
            //fprintf(stderr, "delete human %d - Score %f\n",i+1,humans[i].score);
            humans.erase(humans.begin() + i);
            i--;
        }
    }
}

void ObjectTracking::predict(deque<HumanCluster> &humans) {
    float deltaT = curTimestamp - preTimestamp;

    Matrix A = Matrix(4, 4);
    A << 1 << 0 << deltaT << 0 <<
    0 << 1 << 0 << deltaT <<
    0 << 0 << 1 << 0 <<
    0 << 0 << 0 << 1;

    Matrix Q = Matrix(4, 4);
    Q << 0.16 * deltaT << 0 << 0 << 0 <<
    0 << 0.16 * deltaT << 0 << 0 <<
    0 << 0 << 0.2025 * deltaT << 0 <<
    0 << 0 << 0 << 0.2025 * deltaT;

    for (uint i = 0; i < humans.size(); i++) {
        if (!humans[i].initialized) {
            if (!humans[i].validate()) {
               humans.erase(humans.begin() + i);
               //fprintf(stderr, "delete human %d - Score %f\n",i+1,humans[i].score);
               i--;
               continue;
            }
        }
        humans[i].score -= predict_score * deltaT;
        humans[i].state = A * humans[i].state;
        humans[i].cov = A * humans[i].cov * A.t() + Q;
        humans[i].observe_++;
        humans[i].paired = false;
        humans[i].clust_id_list.push_front(humans[i].clust_id);
        if (humans[i].clust_id_list.size() > 3)
            humans[i].clust_id_list.pop_back();
        humans[i].clust_id = -1;


        //fprintf(stderr, "predict (ID, score, deltaT) = (%d, %.2f, %.2f) \n", humans[i].id, humans[i].score, deltaT);
    }
}

void ObjectTracking::pair(deque<HumanCluster> &humans, deque<Observation> &observations, map<int, int> &pairs) {
    if (humans.size() == 0 || observations.size() == 0) {
       // fprintf(stderr, "No observations");
        return;
    }

    // Matching Matrix
    Matrix MahDis = Matrix(observations.size(), humans.size());
    Matrix EucDis = Matrix(observations.size(), humans.size());

    for (uint i = 0; i < observations.size(); i++) {
        for (uint j = 0; j < humans.size(); j++) {
            // Mahalanobis Distance and Euclidean Distance
            MahDis(i + 1, j + 1) = calculateMahDis(observations[i], humans[j]);
            EucDis(i + 1, j + 1) = calculateEucDis(observations[i], humans[j]);
            //fprintf(stderr, "observation %d, humans %d, minEuc = %.2f, minMah = %.2f \n", i, j, EucDis(i + 1, j + 1),
            //        MahDis(i + 1, j + 1));
        }
    }

    int pairNum = 0;

    //fprintf(stderr, "---------- Pair Start ----------\n");
    while (true) {
        // Get the row and the col of the minimum value in the matrix
        int row = 1;
        int col = 1;

        // int rowE = 0;
        // int colE = 0;
        //float minEuc = 10;

        float minMah = MahDis.Minimum2(row, col);
        float minEuc = EucDis(row, col);

        // float minEuc_2 = EucDis.Minimum2(rowE, colE);
        // float minMah_2 = MahDis(rowE, colE);
        // float minMah = MahDis(row, col);

        // float minMah = MahDis(row, col);
        // If the euclidean distance is different from the mahalanobis one
        // discard the pairing
        //        if (row != rowE || col != colE)
        //        {
        //          if(minEuc > 2.0)
        //            minEuc = 10;
        //        }

        //fprintf(stderr, "row %d, col %d, minEuc = %.2f, minMah = %.2f \n", row, col, minEuc, minMah);

        // Observation index
        int i = row - 1;

        // Human index
        int j = col - 1;

        // Pair observation and human only if distance is smaller than MAX_MAH_DIS and MAX_EUC_DIS
        if (minMah < max_mah_dist) {
            if (minEuc < max_euc_dist) {
                pairs[i] = j;
                pairNum += 1;

                // Puts the row and column selected to infinity to avoid double pairing
                EucDis.Row(row) = std::numeric_limits<float>::infinity();
                EucDis.Column(col) = std::numeric_limits<float>::infinity();
                MahDis.Row(row) = std::numeric_limits<float>::infinity();
                MahDis.Column(col) = std::numeric_limits<float>::infinity();

                //fprintf(stderr, "Success = %d %d, minEuc = %.2f, minMah = %.2f, cov = %.2f %.2f \n",
                //        i, humans[j].id, minEuc, minMah, humans[j].cov(1, 1), humans[j].cov(2, 2));
            }
            else {
                //fprintf(stderr, "Euc Fail -- Hum id: %d Obs: %d, minMah = %.2f - minEuc = %.2f \n", humans[j].id, i,
                //        minMah, minEuc);
                EucDis(row, col) = std::numeric_limits<float>::infinity();
                MahDis(row, col) = std::numeric_limits<float>::infinity();
            }
        }
        else {
            //fprintf(stderr, "Mah Fail = Obs: %d Hum id: %d, minMah = %.2f \n", i, humans[j].id, minEuc);
            break;
        }

        if (pairNum == fmin(observations.size(), humans.size())) {
            break;
        }
    }
    //ROS_INFO("Pair size: %d", pairs.size());
    //fprintf(stderr, "----------  Pair End  ----------\n");
}

void ObjectTracking::update(deque<HumanCluster> &humans, deque<Observation> &observations, map<int, int> &pairs) {
    if (observations.size() == 0) {
        return;
    }

    float deltaT = curTimestamp - preTimestamp;

    Matrix R = Matrix(4, 4);

    R << 0.09 << 0 << 0 << 0 <<
    0 << 0.09 << 0 << 0 <<
    0 << 0 << 0.36 << 0 <<
    0 << 0 << 0 << 0.36;

    deque<bool> unpairs;
    unpairs.resize(observations.size(), true);

    // For Pairing Result
    for (map<int, int>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        // Observation index
        int i = it->first;

        // Human index
        int j = it->second;
        float delta_time = curTimestamp - humans[j].preTimestamp;
        float shape_consistency = 0;
        if (std::max(observations[i].pca, humans[j].pca) / std::min(observations[i].pca, humans[j].pca) > 2)
            shape_consistency += 0.15;
        if (std::max(observations[i].shape, humans[j].shape) / std::min(observations[i].shape, humans[j].shape) > 1.5)
            shape_consistency += 0.15;
        if (humans[j].initialized) {
            observations[i].state(1) =
                    (0.75 - shape_consistency) * observations[i].state(1) + shape_consistency * humans[j].state(1) +
                    0.17 * humans[j].prev_x +
                    0.08 * humans[j].prv2_x;
            observations[i].state(2) =
                    (0.75 - shape_consistency) * observations[i].state(2) + shape_consistency * humans[j].state(2) +
                    0.17 * humans[j].prev_y + 0.08 * humans[j].prv2_y;
        }
        humans[j].length =
                (0.1 + shape_consistency) * humans[j].length + (0.9 - shape_consistency) * observations[i].length;
        humans[j].distance =
                (0.1 + shape_consistency) * humans[j].distance + (0.9 - shape_consistency) * observations[i].distance;
        humans[j].pca = (0.1 + shape_consistency) * humans[j].pca + (0.9 - shape_consistency) * observations[i].pca;
        humans[j].shape =
                (0.1 + shape_consistency) * humans[j].shape + (0.9 - shape_consistency) * observations[i].shape;
        humans[j].base_x = observations[i].base_x;
        humans[j].base_y = observations[i].base_y;
        humans[j].clust_id = observations[i].scan_index;
        humans[j].paired = true;

        float d_x = observations[i].state(1) - humans[j].preState(1);
        float d_y = observations[i].state(2) - humans[j].preState(2);
        float v_x = d_x / delta_time;
        float v_y = d_y / delta_time;
        // if (v_x * v_x + v_y * v_y > 16) {
        //     humans[j].score -= 100;
        //     continue;
        // }
        float dist_thres = 2.7 * delta_time;
        if (d_x > 0.85 * dist_thres) {
            float tao = d_x / dist_thres;
            observations[i].state(1) -= d_x;
            d_x = d_x + pow(2, -tao) * (dist_thres - d_x);
            observations[i].state(1) += d_x;
        } else if (d_x < -dist_thres) {
            float tao = -d_x / dist_thres;
            observations[i].state(1) -= d_x;
            d_x = d_x - pow(2, -tao) * (dist_thres + d_x);
            observations[i].state(1) += d_x;

        }
        if (d_y > 0.85 * dist_thres) {
            float tao = d_y / dist_thres;
            observations[i].state(2) -= d_y;
            d_y = d_y + pow(2, -tao) * (dist_thres - d_y);
            observations[i].state(2) += d_y;
        } else if (d_y < -dist_thres) {
            float tao = -d_y / dist_thres;
            observations[i].state(2) -= d_y;
            d_y = d_y - pow(2, -tao) * (dist_thres + d_y);
            observations[i].state(2) += d_y;
        }
/*
        if (v_x > 2)
            v_x = 2;
        if (v_x < -2)
            v_x = -2;
        if (v_y > 2)
            v_y = 2;
        if (v_y < -2)
            v_y = -2;
  */


        humans[j].prv2_x = humans[j].prev_x;
        humans[j].prv2_y = humans[j].prev_y;
        humans[j].prev_x = observations[i].state(1);
        humans[j].prev_y = observations[i].state(2);
        observations[i].state(3) = d_x / delta_time;

        observations[i].state(4) = d_y / delta_time;


        Matrix H = IdentityMatrix(4);
        Matrix Y = observations[i].state - H * humans[j].state;
        Matrix S = H * humans[j].cov * H.t() + R;
        Matrix K = humans[j].cov * H.t() * S.i();
        Matrix I = IdentityMatrix(K.Ncols());

        // Maintain score if only detected by laser
        humans[j].score += update_score * deltaT;
        if (humans[j].score > 10)
            humans[j].score = 10;
        // Increase score if detected by camera
        /* if (observations[i].camera_detected == true)
         {
           // humans[j].score -= 0.01 * PREDICT_OBJECT_SCORE * deltaT;
           humans[j].score += ((update_score - (calculateEucDis(observations[i], humans[j]) / max_euc_dist)) +
                               (update_score - (calculateMahDis(observations[i], humans[j]) / max_mah_dist)) * 2) * deltaT;
         }
     */
        humans[j].state = humans[j].state + K * Y;
        humans[j].cov = (I - K * H) * humans[j].cov;

        humans[j].preState = humans[j].state;
        humans[j].preTimestamp = curTimestamp;
        humans[j].pair_++;
        unpairs[i] = false;
        //fprintf(stderr, "paired (observation, ID) = (%d, %d) \n", i, humans[j].id);
    }

    // For New Observations Result
    for (uint i = 0; i < observations.size(); i++) {
        // Only the observation detected by image can be put into the human list
        if (unpairs[i] == true && observations[i].camera_detected == true) {
            humans.push_back(
                    HumanCluster(observations[i].base_x, observations[i].base_y, init_id, new_object_score,
                          observations[i].state, R, curTimestamp, observations[i].shape,
                          observations[i].pca, observations[i].length, observations[i].distance, observations[i].prob,
                          observations[i].scan_index));
            init_id += 1;
            if (init_id > 1000)
                init_id = 1;
            pairs[i] = humans.size() - 1;
            // fprintf(stderr, "new (observation, ID) = (%d, %d) \n", i, humans[pairs[i]].id);
        }
    }
}

float ObjectTracking::calculateMahDis(Observation &observation, HumanCluster &human) {
    if (human.cov(1, 1) * human.cov(2, 2) != 0) {
        float sum1 =
                (human.state(1) - observation.state(1)) * (human.state(1) - observation.state(1)) / human.cov(1, 1);
        float sum2 =
                (human.state(2) - observation.state(2)) * (human.state(2) - observation.state(2)) / human.cov(2, 2);
        if (sqrt(sum1 + sum2) > 0)
            return sqrt(sum1 + sum2);
        else
            return 65535;
    }
    else {
        return 65535;
    }
}

float ObjectTracking::calculateEucDis(Observation &observation, HumanCluster &human) {
    float sum1 = (human.state(1) - observation.state(1)) * (human.state(1) - observation.state(1));
    float sum2 = (human.state(2) - observation.state(2)) * (human.state(2) - observation.state(2));

    if (sqrt(sum1 + sum2) > 0)
        return sqrt(sum1 + sum2);
    else
        return 65535;
}
