#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.0175;

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

  //define spreading parameter
  lambda_= 3 - n_x_;

  // initial state vector
   x_ = VectorXd(n_x_);

   // initial covariance matrix
   P_ = MatrixXd(n_x_, n_x_);
   P_.fill(0.0);
   P_.diagonal().fill(1.0);


   //Initialize vector for Unscented Transform weights
   weights_ = VectorXd(2*n_aug_+1);

   // set weights
   weights_(0) = lambda_/(lambda_ + n_aug_);
   double w_i = 1/(2*(lambda_ + n_aug_));
   for(int i=1; i<2*n_aug_ + 1; i++){
     weights_(i) = w_i;
   }

   cout<<"Initial val for P:"<<P_<<endl;


}

UKF::~UKF() {}

void UKF::Init(MeasurementPackage meas_package){
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		/**
		 Convert radar from polar to cartesian coordinates and initialize state.
		 */
		Tools tools;

		VectorXd x_temp(3);
		x_temp << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
		double x = x_temp(0);
		double phi = x_temp(1);
		double phi_dot = x_temp(2);

		// state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
		x_.fill(0.0);
		x_(0) = x * cos(phi);
		x_(1) = x * sin(phi);

		// Extremely rough approximation of the speed (that's better than 0)
		// by projecting radial speed onto x and y axis.
		x_(2) = phi_dot * cos(phi);
		x_(3) = phi_dot * sin(phi);

	} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		/**
		 Initialize state.
		 */
		// set the state with the initial location and zero velocity
		// state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
		x_.fill(0.0);
		x_(0) = meas_package.raw_measurements_[0];
		x_(1) = meas_package.raw_measurements_[1];
	} else {
		/* Provision for adding more sensors in the future */
	}
}
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

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		Init(meas_package);
		is_initialized_ = true;
		// done initializing, no need to predict or update
		previous_timestamp_ = meas_package.timestamp_;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	/**
	 * Update the state transition matrix F according to the new elapsed time.
	 - Time is measured in seconds.
	 * Update the process noise covariance matrix.
	 * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	 */
	double delta_t = (meas_package.timestamp_ - previous_timestamp_)
			/ 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;
	UKF::Prediction(delta_t);

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	/**
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	 */

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		UpdateRadar(meas_package);
	} else {
		// Lidar updates
		UpdateLidar(meas_package);
	}

	return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/** Estimate the object's location. Modify the state
	 vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	MatrixXd Xsig_aug;
	AugmentedSigmaPoints(&Xsig_aug);
	//cout<<"Augmented sigma points, Xsig:\n"<<Xsig_aug<<endl;

	SigmaPointPrediction(Xsig_aug, &Xsig_pred_, delta_t);

	VectorXd x_pred;
	MatrixXd P_out;
	PredictMeanAndCovariance(&x_pred, &P_out);
	//cout<<"Predicted Mean, x:\n"<<x_pred<<endl;
	//cout<<"Predicted Covariance, P:\n"<<P_out<<endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO: Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

	  /**
	    * update the state by using Kalman Filter equations
	  *///measurement matrix
	MatrixXd H_ = MatrixXd(2, 5);
	H_ << 1, 0, 0, 0, 0,
		  0, 1, 0, 0, 0;


	//measurement covariance matrix - laser
	MatrixXd R_laser_ = MatrixXd(2,2);
	R_laser_ << std_laspx_*std_laspx_, 0,
			    0,  std_laspy_*std_laspy_;

	// From predicted state to predicted measurememt
	VectorXd z_pred = H_ * x_;

	/*
	 *Innovation or measurement residual:
	 *Innovation Residual between predicted measurement and actual measurement
	 */
	VectorXd z = meas_package.raw_measurements_;
	VectorXd y = z - z_pred;

	MatrixXd Ht = H_.transpose();
	/* Innovation (or residual) covariance 	 */
	MatrixXd S = H_ * P_ * Ht + R_laser_;

	/* Preparing Kalman gain calculation */
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	/* Kalman gain */
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	const int x_size = 5;
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;


	NIS_laser_ = (z_pred - z).transpose()*Si*(z_pred - z);
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

	//set measurement dimension, radar can measure r, phi, and r_dot
	const int n_z = 3;

	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);


	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z,n_z);
	S.fill(0.0);

	MatrixXd Zsig;
	PredictRadarMeasurement(&z_pred, &S, Zsig);


	VectorXd x_pred;
	MatrixXd P_pred_out;
	//create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z = meas_package.raw_measurements_;
	UpdateState(Zsig, S, z_pred, z, &x_pred, &P_pred_out);

	NIS_radar_ = (z_pred - z).transpose()*S.inverse()*(z_pred - z);

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //cout<<"A = "<<endl<<A<<endl;

  //set sigma points as columns of matrix Xsig
  Xsig.col(0) = x_;

  //calculate sigma points ...
  MatrixXd x_replicated = MatrixXd(n_x_, 2 * n_x_ + 1);
  x_replicated = x_.replicate(1, 5);

  Xsig.block(0, 1, n_x_, n_x_) = x_replicated + sqrt(lambda_ + n_x_) * A;
  Xsig.block(0, 6, n_x_, n_x_) = x_replicated - sqrt(lambda_ + n_x_) * A;

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
  x_aug(5) = 0.0;
  x_aug(6) = 0.0; //mean_a, mean_yawdd = 0

  // Create noise covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_,        0,
       0              ,       std_yawdd_ * std_yawdd_;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, 2, 2) = Q;

  //cout<<"P_aug\n"<<P_aug<<endl;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  MatrixXd x_replicated = x_aug.replicate(1, n_aug_);  // reshape matrix

  Xsig_aug.block(0, 1, n_aug_, n_aug_) = x_replicated  + sqrt(lambda_ + n_aug_) * L;

  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = x_replicated  - sqrt(lambda_ + n_aug_) * L;

  //write result
  *Xsig_out = Xsig_aug;

}

void UKF::SigmaPointPrediction(MatrixXd & Xsig_aug, MatrixXd* Xsig_out, double delta_t) {

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

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

  //write result
  *Xsig_out = Xsig_pred;


}

void UKF::PredictMeanAndCovariance(VectorXd* x_pred_out, MatrixXd* P_out) {

	//predict state mean
	cout<<"Xsig_pred_ rows:"<<Xsig_pred_.rows()<<"  Xsig_pred_ cols:"<<Xsig_pred_.cols()<<endl;
	x_ = (Xsig_pred_ * weights_);

	//predicted state covariance matrix
	P_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		double angle_phi = x_diff(3);
		x_diff(3) = fmod(angle_phi, 2. * M_PI);

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

	//write result
	*x_pred_out = x_;
	*P_out = P_;
}


void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd& Zsig ) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //radar measurement noise standard deviation radius change in m/s
  double std_radrd = 0.1;

  //create matrix for sigma points in measurement space
  cout<<"n_aug_:"<<n_aug_<<endl;

  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

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
      z_pred = z_pred + weights_(i) * Zsig.col(i);
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

    S = S + weights_(i) * z_diff * z_diff.transpose();
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

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
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
  *P_out = P_;
  *x_out = x_;
}

