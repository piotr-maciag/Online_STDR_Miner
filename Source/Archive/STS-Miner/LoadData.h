#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;


struct STPoint
{
	int eventID;
	string eventType;
	double spatialX;
	double spatialY;
	double temporal;
};

struct MCEntry
{
	int clusterID;
	string eventType;
	vector<STPoint> containedInstances;
	int size = 0;
	double spatialXCenter;
	double spatialYCenter;
	double temporalCenter;
};

struct Sequence
{
	int seqID;
	vector<string> sequence;
	vector<STPoint> tailEventSet;
	double seqIndex = 1000.0;
};

struct MCSequence
{
	int seqID;
	vector<string> sequence;
	vector<MCEntry> tailEventSet;
	double seqIndex = 1000.0;
};

struct Cell
{
	int cellID;
	double spatialXbound;
	double spatialYbound;
	double temporalbound;
	vector<STPoint> containedInstances;
};


extern int size;
extern STPoint* data;
extern fstream uchwyt;

extern vector<vector <STPoint>> dataset;
extern vector<vector <STPoint>> sortedDataset;

extern vector<vector<MCEntry>> MCindex;
extern vector<vector<MCEntry>> sortedMCindex;

extern vector<Sequence> SequencesSet;
extern vector<MCSequence> MCSequencesSet;

extern vector<vector<vector<Cell>>> Grid;

extern double threshold;
extern double theta;
extern int K;

void LoadDataset(string);
int CountInstances(string);
void InsertInstance(STPoint);
void TransformData();
void PrintDataset();
void PrintSortedDataset();

void SortDataset();

vector<STPoint> ForwardSweep(vector<STPoint>);
void Miner();
void ExpandSequence(Sequence seq);
double CalculateDR(vector<STPoint> eventSet, vector<STPoint> joinSet, vector<STPoint> eventTypeSet);
void PrintSequences();

bool isEqual(const STPoint &i1, const STPoint &i2);
bool comparisonID(const STPoint &i1, const STPoint &i2);

void ClearSequencesSet();
void ClearDataset();
void ClearStructures();

vector<vector<MCEntry>> ForwardSweep(vector<MCEntry> tailEventSet, vector<MCEntry> indexClusters);
double CalculateDR(vector<MCEntry> tailEventSet, vector<vector<MCEntry>> joinSet, vector<MCEntry> eventTypeSet);

void ClusterData();
void STGrid();
void PrintMCIndex();

void PerformMicroclustering();
void SortMicrocluster();

void PrintSequencesMC();
void InsertIntoSequSet(Sequence seq);



void MinerMC();



