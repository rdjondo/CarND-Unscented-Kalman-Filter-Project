#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

#include "test.h"
#include "ukf.h"

#include <iostream>

using namespace std;


static bool testGenerateSigmaPoints() {

  cout<<"\n#############################"<<endl;
  cout<<endl<<"GenerateSigmaPoints test"<<endl;

  MatrixXd Xsig(5, 11);
  Xsig.fill(0.0);
  UKF ukf = UKF();

  //set example state
  ukf.x_ = VectorXd(ukf.n_x_);
  ukf.x_.fill(0.0);
  ukf.x_ << 5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;

  //set example covariance matrix
  ukf.P_ = MatrixXd(ukf.n_x_, ukf.n_x_);
  ukf.P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
			  -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
			   0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
			  -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
			  -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;



  //re-define spreading parameter as a function of number non-augmentated dimensions of the state vector
  ukf.lambda_= 3 - ukf.n_x_;
  ukf.GenerateSigmaPoints(&Xsig);
  cout << "Xsig :" << endl << Xsig << endl;

  MatrixXd Xsig_expected(ukf.n_x_, 2 * ukf.n_x_ + 1);
	Xsig_expected <<
			5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,
		      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,    1.38,
		    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,  2.2049,
		    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,  0.5015,
		    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879;

  cout << "Xsig_expected:\n" << Xsig_expected << endl;

  double diffNorm = (Xsig_expected - Xsig).norm();
  cout<<"diffNorm"<<endl<<diffNorm<<endl;
  bool test_result =  fabs(diffNorm) < 5e-1;
  if (test_result)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result;
}

static bool testAugmentedSigmaPoints() {

  cout<<"\n#############################"<<endl;
  cout<<"\n"<<"AugmentedSigmaPoints test"<<endl;

  UKF ukf;

  //Process noise standard deviation longitudinal acceleration in m/s^2
  ukf.std_a_ = 0.2;

  //Process noise standard deviation yaw acceleration in rad/s^2
  ukf.std_yawdd_ = 0.2;


  //set example state
  ukf.x_ = VectorXd(ukf.n_x_);
  ukf.x_ << 5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  //create example covariance matrix
  ukf.P_ = MatrixXd(ukf.n_x_, ukf.n_x_);
  ukf.P_ <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
             0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  MatrixXd Xsig_aug;

  //re-define spreading parameter as a function of number non-augmentated dimensions of the state vector
  ukf.lambda_= 3 - ukf.n_aug_;
  ukf.AugmentedSigmaPoints(&Xsig_aug);
    cout<<Xsig_aug<<endl << Xsig_aug << endl;


  /* expected result:
     Xsig_aug =
  5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441
    1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38
  2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049
  0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015
  0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528
     0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0
     0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641
  */

  MatrixXd Xsig_expected = MatrixXd(ukf.n_aug_ , 2 * ukf.n_aug_ + 1);

  Xsig_expected <<
      5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
      2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
      0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
      0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
           0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
           0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;
  cout << "\nXsig_expected:\n" << Xsig_expected << endl;

  bool test_result = (Xsig_expected - Xsig_aug).norm() < 1e-5;
  if (test_result)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result;
}


static bool testSigmaPointPrediction() {

  cout<<"\n#############################"<<endl;
  cout<<"\n"<<"SigmaPointPrediction test"<<endl;

  UKF ukf;

  //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(ukf.n_aug_, 2 * ukf.n_aug_ + 1);
  Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  //call function under test
  MatrixXd Xsig_pred(ukf.n_x_, 2 * ukf.n_aug_ + 1);

  double delta_t = 0.1;  //compute time diff in sec
  ukf.SigmaPointPrediction(Xsig_aug, &Xsig_pred, delta_t);

  cout << "\nXsig_pred:\n" << Xsig_pred << endl;

  MatrixXd Xsig_pred_expected(ukf.n_x_, ukf.n_aug_* 2 + 1);
  Xsig_pred_expected <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  double diffNorm = (Xsig_pred_expected - Xsig_pred).norm();
  cout<<"diffNorm:"<<endl<<diffNorm<<endl;
  bool test_result =  diffNorm < 1e-1;
  if (test_result)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result;
}


static bool testPredictMeanAndCovariance() {

  cout<<"\n#############################"<<endl;
  cout<<"\n"<<"PredictMeanAndCovariance test"<<endl;

  UKF ukf;

  VectorXd x_out;
  MatrixXd P_out;

  //create example matrix with predicted sigma points
  ukf.Xsig_pred_ = MatrixXd(ukf.n_x_, 2 * ukf.n_aug_ + 1);
  ukf.Xsig_pred_ <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;


  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(ukf.n_x_, ukf.n_x_);


  ukf.PredictMeanAndCovariance(&x_out, &P_out);


  //create expected matrix for predicted state covariance
  MatrixXd P_expected = MatrixXd(ukf.n_x_, ukf.n_x_);
  P_expected <<
  0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
  -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
  0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
 -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
 -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;

  //create expected vector for predicted state mean
  VectorXd x_expected = VectorXd(ukf.n_x_);
  x_expected <<
     5.93637,
     1.49035,
     2.20528,
    0.536853,
    0.353577;

  double diffPNorm = (P_expected - ukf.P_).norm();
  cout<<"diffPNorm"<<endl<<diffPNorm<<endl;
  bool test_result_P =  diffPNorm < 1e-1;

  double diffXNorm = (x_expected - x_out).norm();
  cout<<"diffXNorm"<<endl<<diffXNorm<<endl;
  bool test_result_x =  diffXNorm < 1e-1;
  if (test_result_x)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result_P && test_result_x;
}

static bool testPredictRadarMeasurement(){

  cout<<"\n#############################"<<endl;
  cout<<"\n"<<"PredictRadarMeasurement test"<<endl;

  //set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z = 3;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //measurement covariance matrix S
  MatrixXd S_out = MatrixXd(n_z,n_z);
  S_out.fill(0.0);

  UKF ukf;



  //radar measurement noise standard deviation radius in m
  ukf.std_radr_ = 0.3;

  //radar measurement noise standard deviation angle in rad
  ukf.std_radphi_ = 0.0175;


  //create example matrix with predicted sigma points
  ukf.Xsig_pred_ = MatrixXd(ukf.n_x_, 2 * ukf.n_aug_ + 1);
  ukf.Xsig_pred_.fill(0.0);
  ukf.Xsig_pred_ <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  MatrixXd Zsig;
  ukf.PredictRadarMeasurement(&z_pred, &S_out, Zsig);

  VectorXd  z_pred_expected= VectorXd(n_z);
  z_pred_expected << 6.12155,
      0.245993,
      2.10313;

  MatrixXd S_expected = MatrixXd(n_z,n_z);
  S_expected << 0.0946171, -0.000139448, 0.00407016,
      -0.000139448, 0.000617548, -0.000770652,
      0.00407016, -0.000770652, 0.0180917;

  double diffZNorm = (z_pred_expected - z_pred).norm();
  cout<<"diffZNorm"<<endl<<diffZNorm<<endl;
  bool test_result_x =  diffZNorm < 1e-2;
  if (test_result_x)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  double diffSNorm = (S_expected - S_out).norm();
  cout<<"diffSNorm"<<endl<<diffSNorm<<endl;
  test_result_x &=  diffSNorm < 1e-1;
  if (test_result_x)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result_x;
}


static bool testUpdateState() {

  cout<<"\n#############################"<<endl;
  cout<<"\n"<<"UpdateState test"<<endl;

  UKF ukf;

  //set state dimension
  ukf.n_x_ = 5;

  //set augmented dimension
  ukf.n_aug_ = 7;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  ukf.lambda_ = 3 - ukf.n_aug_;

  //create example matrix with predicted sigma points
  ukf.Xsig_pred_ = MatrixXd(ukf.n_x_, 2 * ukf.n_aug_ + 1);
  ukf.Xsig_pred_ <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  //create example vector for predicted state mean
  ukf.x_ = VectorXd(ukf.n_x_);
  ukf.x_ <<
     5.93637,
     1.49035,
     2.20528,
    0.536853,
    0.353577;

  //create expected matrix for predicted state covariance
  ukf.P_ = MatrixXd(ukf.n_x_, ukf.n_x_);
  ukf.P_  <<
  0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
  -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
  0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
 -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
 -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;



  //create example matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * ukf.n_aug_ + 1);
  Zsig <<
      6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
     0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
      2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

  //create example vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred <<
      6.12155,
     0.245993,
      2.10313;

  //create example matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z,n_z);
  S <<
      0.0946171, -0.000139448,   0.00407016,
   -0.000139448,  0.000617548, -0.000770652,
     0.00407016, -0.000770652,    0.0180917;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z <<
      5.9214,
      0.2187,
      2.0062;


  VectorXd x_out(ukf.n_x_) ;
  MatrixXd P_out(ukf.n_x_,ukf.n_x_);
  //x_out = ukf.x_;
  //P_out = ukf.P_;
  ukf.UpdateState(Zsig, S, z_pred, z, &x_out, &P_out);

  //create expected vector for predicted state mean
  VectorXd x_expected(ukf.n_x_) ;
  x_expected <<
       5.92276,
       1.41823,
       2.15593,
      0.489274,
      0.321338;

  MatrixXd P_expected(ukf.n_x_,ukf.n_x_);
  P_expected <<
     0.00361579, -0.000357881,   0.00208316, -0.000937196,  -0.00071727   ,
     -0.000357881,   0.00539867,   0.00156846,   0.00455342,   0.00358885 ,
       0.00208316,   0.00156846,   0.00410651,   0.00160333,   0.00171811 ,
     -0.000937196,   0.00455342,   0.00160333,   0.00652634,   0.00669436 ,
      -0.00071719,   0.00358884,   0.00171811,   0.00669426,   0.00881797 ;


  double diffPNorm = (P_expected - P_out).norm();
  cout<<"diffPNorm"<<endl<<diffPNorm<<endl;
  bool test_result =  diffPNorm < 1e-2;

  double diffXNorm = (x_expected - x_out).norm();
  cout<<"diffXNorm"<<endl<<diffXNorm<<endl;
  test_result &=  diffXNorm < 1e-2;
  if (test_result)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result;
}

static bool integrateSteps(){

  cout<<"\n#############################"<<endl;
  cout<<endl<<"integrateSteps test"<<endl;

  UKF ukf;

  //set example state
  ukf.x_ = VectorXd(ukf.n_x_);
  ukf.x_ <<   5.7441,
           1.3800,
           2.2049,
           0.5015,
           0.3528;

  //set example covariance matrix
  ukf.P_ = MatrixXd(ukf.n_x_, ukf.n_x_);
  ukf.P_ <<
         0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
        -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
         0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
        -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
        -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  MatrixXd Xsig(5, 11);
  Xsig.fill(0.0);
  ukf.GenerateSigmaPoints(&Xsig);

  MatrixXd Xsig_aug;
  ukf.AugmentedSigmaPoints(&Xsig_aug);

	double delta_t = 0.1;  //compute time diff in sec
	ukf.SigmaPointPrediction(Xsig_aug, &ukf.Xsig_pred_, delta_t);

	VectorXd x_pred;
	MatrixXd P_pred_out;
	ukf.PredictMeanAndCovariance(&x_pred, &P_pred_out);

	//set measurement dimension, radar can measure r, phi, and r_dot
	const int n_z = 3;

	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);

	MatrixXd Zsig;
	ukf.PredictRadarMeasurement(&z_pred, &S, Zsig);

	//create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z <<
      5.9214,
      0.2187,
      2.0062;
	ukf.UpdateState(Zsig, S, z_pred, z, &x_pred, &P_pred_out);

	//create expected vector for predicted state mean
	VectorXd x_expected(ukf.n_x_) ;
  x_expected <<
       5.92276,
       1.41823,
       2.15593,
      0.489274,
      0.321338;

  MatrixXd P_expected(ukf.n_x_,ukf.n_x_);
  P_expected <<
     0.00361579, -0.000357881,   0.00208316, -0.000937196,  -0.00071727   ,
     -0.000357881,   0.00539867,   0.00156846,   0.00455342,   0.00358885 ,
       0.00208316,   0.00156846,   0.00410651,   0.00160333,   0.00171811 ,
     -0.000937196,   0.00455342,   0.00160333,   0.00652634,   0.00669436 ,
      -0.00071719,   0.00358884,   0.00171811,   0.00669426,   0.00881797 ;


  double diffPNorm = (P_expected - P_pred_out).norm();
  cout<<"diffPNorm"<<endl<<diffPNorm<<endl;
  bool test_result =  diffPNorm < 1e-2;

  double diffXNorm = (x_expected - x_pred).norm();
  cout<<"diffXNorm"<<endl<<diffXNorm<<endl;
  test_result &=  diffXNorm < 1e-1;
  if (test_result)   cout << "PASS"<<endl;
  else cout << "FAIL"<<endl;

  return test_result;
}



void test() {
  assert(testGenerateSigmaPoints()==true);
  assert(testAugmentedSigmaPoints()==true);
  assert(testSigmaPointPrediction()==true);
  assert(testPredictMeanAndCovariance()==true);
  assert(testPredictRadarMeasurement()==true);
  assert(testUpdateState()==true);
  assert(integrateSteps()==true);
}
