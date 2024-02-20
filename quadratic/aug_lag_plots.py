import sys
sys.path.insert(0, '../utilities')
from plot_utils import plot_aug_lag
from quad_prob_def import quad_prob

Q,A,b,theta_star,lambda_star = quad_prob()

plot_aug_lag(Q,A,b,0,theta_star,lambda_star)
plot_aug_lag(Q,A,b,1,theta_star,lambda_star)
plot_aug_lag(Q,A,b,2,theta_star,lambda_star)
