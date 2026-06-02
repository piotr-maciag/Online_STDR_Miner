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
	//1exp N 50,100 ... 250 crimes5000,crimes10000
	//2exp N 100, 200, 300, ... 1200, crimes10000
	//3exp theta 0.7 .. 0.2, 0.05 crimes3000

	/*fstream exp3ExecTime;
	//exp3ExecTime.open("dataResultsTheta.txt", fstream::out);

	for(theta = 0.7; theta >= 1.15; theta -= 0.05)
	{
	string path = "crimes3000.txt";
	LoadDataset(path);
	TransformData();
	SortDataset();
	//PrintSortedDataset();

	//theta = 0.7;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	MinerSPTree();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = duration_cast<seconds>( t2 - t1 ).count();
	cout << theta << " " <<duration << endl;

	exp3ExecTime << duration << endl;
	ClearStructures();
	//theta = 0.00;
	}

	exp3ExecTime.close();

	exp3ExecTime.open("dataResultsNaiveTheta2.txt", fstream::out);

	for(theta = 0.2; theta >= 0.15; theta -= 0.05)
	{
	string path = "crimes3000.txt";
	LoadDataset(path);
	TransformData();
	SortDataset();
	//PrintSortedDataset();

	//theta = 0.7;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	Miner();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = duration_cast<seconds>( t2 - t1 ).count();
	cout << theta << " " << duration << endl;

	exp3ExecTime << duration << endl;
	ClearStructures();
	//theta = 0.00;
	}

	exp3ExecTime.close();*/


	/*string path = "crimes.txt";

		LoadDataset(path);
		TransformData();
		SortDataset();
		//PrintSortedDataset();
		cout << "DataLoaded" << endl;
		//theta = 0.7;

		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		MinerSPTree();
		high_resolution_clock::time_point t2 = high_resolution_clock::now();

		PrintSequencesTopSPTree();
		ClearStructures();
	 */


	fstream exp3ExecTime;

	string paths[] = {"crimes.txt", "crimes5000.txt"};
	for(R = 300; R <= 300; R += 300) //1. 300 m,
	{
	for(T = 11520; T <= 11520; T+= 14400) //1. 300m, 5760 (4 dni); 2. 300m, 11520 (8 dni); 3. 200m, 14400;
	{
	for (int i = 0; i < 1; i++)
	{
		fstream exp3ExecTime;
		exp3ExecTime.open("dataResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

		cout << "dataResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;

		//1. for(theta = 0.1; theta >= 0.01; theta -= 0.005)
		//2. for(theta = 0.5; theta >= 0.1; theta -= 0.05)
		//3. for(theta = 0.08; theta >= 0.032; theta -= 0.005)
		for(theta = 0.55; theta >= 0.12; theta -= 0.05)
		{
		string path = paths[i];
		LoadDataset(path);
		TransformData();
		SortDataset();
		//PrintSortedDataset();

		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		MinerSPTree();
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		int NumPatterns = CountPatterns();

		auto duration = duration_cast<seconds>( t2 - t1 ).count();
		cout << theta << " " <<duration << " " << NumPatterns << endl;

		exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
		ClearStructures();
		}

		exp3ExecTime.close();

		exp3ExecTime.open("dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

		cout << "dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;
		for(theta = 0.55; theta >= 0.12; theta -= 0.05)
		{
		string path = paths[i];
		LoadDataset(path);
		TransformData();
		SortDataset();

		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		MinerSPTreeClosed();
		high_resolution_clock::time_point t2 = high_resolution_clock::now();

		int NumPatterns = CountPatterns();
		auto duration = duration_cast<seconds>( t2 - t1 ).count();
		cout << theta << " " <<duration << " " << NumPatterns << endl;

		exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
		ClearStructures();
		}

		exp3ExecTime.close();
		}
		}
		}

	/*
	string paths[] = {"crimes_modified_3.txt"};

	for(R = 600; R <= 600; R += 200) //1. 300 m,
	{
		for(T = 28800; T <= 28800; T+= 14400) //1. 300m, 5760 (4 dni); 2. 300m, 11520 (8 dni); 3. 200m, 14400;
		{
			for (int i = 0; i < 1; i++)
			{
				fstream exp3ExecTime;
				exp3ExecTime.open("TeSame_ResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

				cout << "dataResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;

				//1. for(theta = 0.1; theta >= 0.01; theta -= 0.005)
				//2. for(theta = 0.5; theta >= 0.1; theta -= 0.05)
				//3. for(theta = 0.08; theta >= 0.032; theta -= 0.005)
				for(theta = 0.01; theta >= 0.001; theta -= 0.001)
				{
					string path = paths[i];
					LoadDataset(path);
					TransformData();
					SortDataset();
					//PrintSortedDataset();

					high_resolution_clock::time_point t1 = high_resolution_clock::now();
					MinerSPTree();
					high_resolution_clock::time_point t2 = high_resolution_clock::now();
					int NumPatterns = CountPatterns();

					auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
					cout << theta << " " <<duration << " " << NumPatterns << endl;

					exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
					ClearStructures();
				}

				exp3ExecTime.close();

				exp3ExecTime.open("TeSame_ dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

				cout << "dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;
				for(theta = 0.01; theta >= 0.001; theta -= 0.001)
				{
					string path = paths[i];
					LoadDataset(path);
					TransformData();
					SortDataset();

					high_resolution_clock::time_point t1 = high_resolution_clock::now();
					MinerSPTreeClosed();
					high_resolution_clock::time_point t2 = high_resolution_clock::now();

					int NumPatterns = CountPatterns();
					auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
					cout << theta << " " <<duration << " " << NumPatterns << endl;

					exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
					ClearStructures();
				}

				exp3ExecTime.close();
			}
		}
	}
*/
	/*
	string path = "crimes_modified_3.txt";

			LoadDataset(path);
			TransformData();
			SortDataset();
			//PrintSortedDataset();
			cout << "DataLoaded" << endl;
			theta = 0.01;
			R = 500;
			T = 43200;
			MinerSPTreeClosed();

			PrintSequences();
			cout << CountPatterns();
			ClearStructures();

	*/


	//Synthetic results
/*
	string paths[] = {"DataSe1Ps20Pn20Ni10Nf5.txt"};

		for(R = 10; R <= 10; R += 10)
		{
			for(T = 10; T <= 10; T+= 10)
			{
				for (int i = 0; i < 1; i++)
				{
					fstream exp3ExecTime;
					exp3ExecTime.open("Tesame_dataResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

					cout << "dataResultsThetaNonClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;

					//1. DataSe1Ps20Pn20Ni10Nf5.txt, for(theta = 0.01; theta >= 0.0015; theta -= 0.0005)
					for(theta = 0.01; theta >= 0.0015; theta -= 0.0005)
					{
						string path = paths[i];
						LoadDataset(path);
						TransformData();
						SortDataset();
						//PrintSortedDataset();

						high_resolution_clock::time_point t1 = high_resolution_clock::now();
						MinerSPTree();
						high_resolution_clock::time_point t2 = high_resolution_clock::now();
						int NumPatterns = CountPatterns();

						auto duration = duration_cast<seconds>( t2 - t1 ).count();
						cout << theta << " " <<duration << " " << NumPatterns << endl;

						exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
						ClearStructures();
					}

					exp3ExecTime.close();

					exp3ExecTime.open("Tesame_dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt", fstream::out);

					cout << "dataResultsThetaClosedR" + to_string(R) + "_T" + to_string(T) + "_" + paths[i] + ".txt" << endl;
					for(theta = 0.01; theta >= 0.0016; theta -= 0.0005)
					{
						string path = paths[i];
						LoadDataset(path);
						TransformData();
						SortDataset();

						high_resolution_clock::time_point t1 = high_resolution_clock::now();
						MinerSPTreeClosed();
						high_resolution_clock::time_point t2 = high_resolution_clock::now();

						int NumPatterns = CountPatterns();
						auto duration = duration_cast<seconds>( t2 - t1 ).count();
						cout << theta << " " <<duration << " " << NumPatterns << endl;

						exp3ExecTime << theta << " " <<duration << " " << NumPatterns << endl;
						ClearStructures();
					}

					exp3ExecTime.close();
				}
			}
		}

*/
/*
	string path = "example.txt";

	LoadDataset(path);
	TransformData();
	SortDataset();
			//PrintSortedDataset();
	cout << "DataLoaded" << endl;
	theta = 0.2;
	R = 10;
	T = 20;
	MinerSPTreeClosed();

	PrintSequences();
	cout << CountPatterns();
	ClearStructures();
*/
}
