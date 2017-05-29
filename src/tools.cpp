#include <vector>
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::endl;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if(estimations.size() != ground_truth.size()){
		cout << "Estimation and ground_truth data sizes must be the same" << endl;
		return rmse;
	} else if (estimations.size() == 0){
		cout << "Error: estimation size is 0" << endl;
		return rmse;
	}

	// sum squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array(); // Why call to array() member function
		rmse += residual;
	}

	//calculate the average the squared residuals
	rmse = rmse/estimations.size();

	//calculate the squared root of the average
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}
