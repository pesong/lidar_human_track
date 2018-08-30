/*****************************************************************
 *
 * This file is part of the People2D project
 *
 * People2D Copyright (c) 2011 Luciano Spinello
 *
 * This software is licensed under the "Creative Commons
 * License (Attribution-NonCommercial-ShareAlike 3.0)"
 * and is copyrighted by Luciano Spinello
 *
 * Further information on this license can be found at:
 * http://creativecommons.org/licenses/by-nc-sa/3.0/
 *
 * People2D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.
 *
 *****************************************************************/
#ifndef LIBFEATURES_H
#define LIBFEATURES_H

#include <vector>
#include <string>
#include <algorithm>

#include <lgeometry.hpp>

#define L_MINCLUSTERSZ 3
#define L_EPSNUM 1e-10
#define L_HUGENUM 1e10

#define FEATURE_SET_0 17
#define FEATURE_SET_1 63
#define FEATURE_BASIS 13

typedef float Real;

class lfeatures_class
{
	private:
	 	std::vector <std::string> feature_descr;
		void descr(void);
		int feature_01(Point3D_container *laserfeat_cluster, Real *out);
		int feature_02(Point3D_container *laserfeat_cluster, Real *out);
		int feature_03(Point3D_container *laserfeat_cluster, Real *out);
		int feature_04(Point3D_container *laserfeat_cluster, Real *out);
		int feature_05(Point3D_container *laserfeat_cluster, Real *out);
		int feature_06(Point3D_container *laserfeat_cluster, Real *out);
		int feature_07(Point3D_container *laserfeat_cluster, Real *out);
		int feature_08(Point3D_container *laserfeat_cluster, Real *out);
		int feature_09(Point3D_container *laserfeat_cluster, Real *out);
		int feature_10(Point3D_container *laserfeat_cluster, Real *out);
		int feature_11(Point3D_container *laserfeat_cluster, Real *out);
		int feature_12(Point3D_container *laserfeat_cluster, Real *out);
		int feature_13(Point3D_container *laserfeat_cluster, Real *out);
		int feature_14(Point3D_container *laserfeat_cluster, Real *out);
		int feature_15(Point3D_container *laserfeat_cluster, Real *out);
		int feature_16(Point3D_container *laserfeat_cluster, Real *out);
        int feature_01_comparative(std::vector <Point3D_container> &all_laserfeat_cluster, uint curidx, Real *out);

		int feature_size;
		int feature_set;

		#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
		typedef int  (lfeatures_class::*ptr_feature)(Point3D_container*, Real*);
        std::vector<ptr_feature> ptr_feature_vector;

		#define CALL_MEMBER_FN_CMP(object,ptrToMember)  ((object).*(ptrToMember))
        typedef int  (lfeatures_class::*ptr_feature_cmp)(std::vector<Point3D_container> &, uint, Real*);
        std::vector<ptr_feature_cmp> ptr_feature_vector_cmp;

	public:
		lfeatures_class(int feature_set);
        void compute_descriptor(std::vector <Point3D_container> &all_laserfeat_cluster, std::vector< std::vector<Real> > &descriptor);

};
#endif
