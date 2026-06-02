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

struct Child
{
	string eventType;
	vector<STPoint> tailEventSet;
};

struct Sequence
{
	int seqID;
	vector<string> sequence;
	vector<vector<STPoint *>> tailEventSet;

	vector<STPoint> lastTailEventSet;
	string lastEventType;
	double PI = 1000.0;

	Sequence* firstParent = NULL;
	Sequence* secondParent = NULL;

	vector<Sequence*> children;

	bool isClosed = true;
};



struct Node
{
	int ID;
	string eventType;
	STPoint tailEventSet;

	Node* pointer;
	vector<Node*> children;
};

struct SPTreeStruct
{
	vector<Node*> L1;

};

extern int size;
extern STPoint* data;
extern fstream uchwyt;

extern vector<vector <STPoint>> dataset;
extern vector<vector <STPoint>> sortedDataset;

extern vector<vector<Sequence>> SequencesSet;

extern double threshold;
extern double theta;
extern int K;
extern int N;

extern double R;
extern double T;

void LoadDataset(string);
int CountInstances(string);
void InsertInstance(STPoint);
void TransformData();
void PrintDataset();
void PrintSortedDataset();

void SortDataset();

vector<STPoint> ForwardSweep(vector<STPoint>);
void Miner();
void MinerSPTree();
void MinerSPTreeClosed();
void ExpandSequence(Sequence seq);
double CalculateDR(vector<STPoint> eventSet, vector<STPoint> joinSet, vector<STPoint> eventTypeSet);
void PrintSequences();
void PrintSequencesTop();
void PrintSequencesTopSPTree();
int CountPatterns();

//bool isEqual(const STPoint &i1, const STPoint &i2);
//bool comparisonID(const STPoint &i1, const STPoint &i2);

void ClearSequencesSet();
void ClearDataset();
void ClearStructures();

void InsertIntoSequSet(Sequence seq);




