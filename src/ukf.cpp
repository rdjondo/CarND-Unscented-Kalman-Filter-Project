#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

static void print(string text, MatrixXd& X){
	cout<<text<<":\n"<<X<<endl;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

    /*
    VectorXd x_pred = VectorXd(5);
    MatrixXd P_pred = MatrixXd(5, 5);
    ukf.PredictMeanAndCovariance(&x_pred, &P_pred); */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  /*******************************************************************************
   * Student part begin
   ******************************************************************************/

  //your code goes here
  //calculate sigma points ...
  Xsig.col(0) = x_;
  MatrixXd x_replicated = x_.replicate(1, 5);
  MatrixXd Unsc_plus = x_replicated + sqrt(lambda_ + n_x_) * A;
  MatrixXd Unsc_minus = x_replicated - sqrt(lambda_ + n_x_) * A;

  //set sigma points as columns of matrix Xsig
  Xsig.block(0, 1, n_x_, n_x_) = Unsc_plus;
  Xsig.block(0, 6, n_x_, n_x_) = Unsc_minus;

  //write result
  *Xsig_out = Xsig;

}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0; //mean_a, mean_yawdd = 0

  // Create noise covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_, 0,
      0, std_yawdd_ * std_yawdd_;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, 2, 2) = Q;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  MatrixXd x_replicated = x_aug.replicate(1, n_aug_);  // reshape matrix

  Xsig_aug.block(0, 1, n_aug_, n_aug_) = x_replicated
      + sqrt(lambda_ + n_aug_) * L;

  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = x_replicated
      - sqrt(lambda_ + n_aug_) * L;

  //write result
  *Xsig_out = Xsig_aug;

}

void UKF::SigmaPointPrediction(MatrixXd & Xsig_aug, MatrixXd* Xsig_out) {

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  double delta_t = 0.1;  //TODO: compute time diff in sec

  /*******************************************************************************
   * Student part begin
   ******************************************************************************/

  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  /*******************************************************************************
   * Student part end
   ******************************************************************************/

  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;


}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

  //create vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);

  // set weights
  weights(0) = lambda_/(lambda_ + n_aug_);
  double w_i = 1/(2*(lambda_ + n_aug_));
  for(int i=1; i<2*n_aug_ + 1; i++){
    weights(i) = w_i;
  }

  //predict state mean
  x_ = (Xsig_pred_*weights);

  //predicted state covariance matrix
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3)=fmod(x_diff(3), 2.*M_PI);

    P_ = P_ + weights(i) * x_diff * x_diff.transpose() ;
  }


/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;

  //write result
  *x_out = x_;
  *P_out = P_;
}


void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;


  //set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
   double weight_0 = lambda_/(lambda_+n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }


  //radar measurement noise standard deviation radius change in m/s
  double std_radrd = 0.1;

  //create matrix for sigma points in measurement space
  cout<<"n_aug_:"<<n_aug_<<endl;

  //MatrixXd Zsig(n_z, 2 * n_aug_ + 1);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  //print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateState(MatrixXd& Zsig, MatrixXd& S, VectorXd& z_pred,
                      VectorXd& z, VectorXd* x_out, MatrixXd* P_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z = 3;

  //set vector for weights
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights(i) = weight;
  }

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //print result
  std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

  //write result
  *x_out = x_;
  *P_out = P_;

  /* expected result x:
   x =
   5.92276
   1.41823
   2.15593
   0.489274
   0.321338
   */

  /* expected result P:
   P =
   0.00361579 -0.000357881   0.00208316 -0.000937196  -0.00071727
   -0.000357881   0.00539867   0.00156846   0.00455342   0.00358885
   0.00208316   0.00156846   0.00410651   0.00160333   0.00171811
   -0.000937196   0.00455342   0.00160333   0.00652634   0.00669436
   -0.00071719   0.00358884   0.00171811   0.00669426   0.00881797
   */

}

