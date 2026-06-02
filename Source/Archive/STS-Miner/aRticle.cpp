//============================================================================
// Name        : aRticle.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <ctime>
#include <chrono>

#include "LoadData.h"
using namespace std;
using namespace std::chrono;


int main()
{

	int series_num = 1;

	fstream exp3ExecTime;
	exp3ExecTime.open("dataResults5000.txt", fstream::out);

	double execTime = 0.0;

	//for(int Nf = 15; Nf <= 25; Nf += 10)
	{
		//for(int ps = 2; ps <= 10; ps += 1)
		{
			//for(K = 50; K <= 250; K += 50)
			{
				//for(int series = 1; series <= series_num; series++)
				{
				K = 1000;
				string path = "crimes5000.txt";
				cout << "Dataset " << path  << " " << K << " " << endl;

				LoadDataset(path);
				TransformData();
				SortDataset();



				//PrintSortedDataset();

				high_resolution_clock::time_point t1 = high_resolution_clock::now();
				Miner();
				high_resolution_clock::time_point t2 = high_resolution_clock::now();
				auto duration = duration_cast<seconds>( t2 - t1 ).count();

				//execTime += duration;
				cout << duration << endl;
				exp3ExecTime << duration << endl;
				//PrintSequences();
				ClearStructures();
				theta = 1.0;
				}

				//exp3ExecTime << execTime/(double)series_num << "\t";
				execTime = 0.0;

			}

			exp3ExecTime << endl;
		}

		exp3ExecTime << endl << "#########" << endl;
	}

	exp3ExecTime.close();
}
