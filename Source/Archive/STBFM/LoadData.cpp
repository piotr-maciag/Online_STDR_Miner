#include "LoadData.h"

double T;
double R;



double Dsize = 1000;
double Tsize = 1200;

//STGrid Parameters

double spatialTr = 20.0; //for diameter
double temporalTr = 20.0; //for diameter

double threshold;

//########################################

double theta;
int K;
int N = 250;

int size;
STPoint* data;
fstream uchwyt;

int numOfEventTypes; // not used

vector<vector <STPoint>> dataset;
vector<vector <STPoint>> sortedDataset;

vector<vector<Sequence>> SequencesSet;
vector<Sequence> Top;

vector<vector<Sequence*>> SequencesSetSPTree;
vector<Sequence*> TopSPTree;

int GseqID = 0;



bool comparison(const STPoint &i1, const STPoint &i2)
{
	return i1.temporal < i2.temporal;
}

bool comparisonPT(STPoint* i1, STPoint* i2)
{
	return i1->temporal < i2->temporal;
}

bool comparisonID(const STPoint &i1, const STPoint &i2)
{
	return i1.eventID < i2.eventID;
}

bool comparisonIDPT(STPoint* i1, STPoint* i2)
{
	return i1->eventID < i2->eventID;
}

bool isEqual(const STPoint &i1, const STPoint &i2)
{
	return (i1.eventID == i2.eventID);
}

bool isNotEqual(const STPoint &i1, const STPoint &i2)
{
	return (i1.eventID != i2.eventID);
}

void SortDataset()
{
	sortedDataset = dataset;
	for(int i = 0; i < dataset.size(); i++)
	{
		sort(sortedDataset[i].begin(), sortedDataset[i].end(), comparison);
	}
}


#define earthRadiusKm 6371.0

// This function converts decimal degrees to radians
double deg2rad(double deg) {
  return (deg * M_PI / 180);
}

//  This function converts radians to decimal degrees
double rad2deg(double rad) {
  return (rad * 180 / M_PI);
}

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @return The distance between the two points in kilometers
 */
double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
  double lat1r, lon1r, lat2r, lon2r, u, v;
  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  u = sin((lat2r - lat1r)/2);
  v = sin((lon2r - lon1r)/2);
  return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v)) * 1000;
}

double distanceSpatial(double pSpatialY, double pSpatialX, double qSpatialY, double qSpatialX)
{
	return sqrt((pSpatialX - qSpatialX)*(pSpatialX - qSpatialX) + (pSpatialY - qSpatialY)*(pSpatialY - qSpatialY));
}

vector<STPoint *> ForwardSweep(vector<STPoint *> tailEventSet, vector<STPoint *> instancesSet)
		{

	sort((tailEventSet.begin()), (tailEventSet.end()), comparisonPT);
	sort(instancesSet.begin(), instancesSet.end(), comparisonPT);
	vector<STPoint*> joinResult;



	while(tailEventSet.empty() != true && instancesSet.empty() != true)
	{
		int pindex = 0, qindex = 0;

		STPoint* p = tailEventSet[pindex];
		STPoint* q = instancesSet[qindex];

		if(p->temporal < q->temporal)
		{
			tailEventSet.erase(tailEventSet.begin());
			while(p->temporal + T > q->temporal)
			{

				double dist = distanceEarth(p->spatialY, p->spatialX, q->spatialY, q->spatialX);

				if(dist <= R)
				{
					joinResult.push_back(q);
				}
				if(qindex < instancesSet.size() - 1)
				{
					qindex++;
					q = instancesSet[qindex];
				}
				else
				{
					break;
				}
			}
		}
		else
		{
			instancesSet.erase(instancesSet.begin());
		}
	}

	std::vector<int>::iterator it;
//	sort( joinResult.begin(), joinResult.end(), comparisonID);
	//joinResult.erase(unique(joinResult.begin(), joinResult.end(), isEqual), joinResult.end());

	sort(joinResult.begin(), joinResult.end(), comparisonIDPT);
	joinResult.erase(unique(joinResult.begin(), joinResult.end()), joinResult.end());


	return joinResult;
		}

////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

double CalculatePI(vector<STPoint *> instSet)
{
	double totalNumber;

	if(instSet.empty() == true)
	{
		return 0;
	}

	for(int i = 0; i < sortedDataset.size(); i++)
	{
		if(instSet[0]->eventType == sortedDataset[i][0].eventType)
		{
			totalNumber = sortedDataset[i].size();
		}
	}


	return (double(instSet.size())/totalNumber);
}

void InsertIntoSequSet(Sequence seq)
{
	if(Top.empty() == true)
	{
		Top.push_back(seq);
		return;
	}

	for(int i = 0; i < Top.size(); i++)
	{
		if(Top[i].PI < seq.PI)
		{
			Top.insert(Top.begin() + i, seq);
			return;
		}
	}

	Top.push_back(seq);
}

void InsertIntoSequSet(Sequence* seq)
{
	if(TopSPTree.empty() == true)
	{
		TopSPTree.push_back(seq);
		return;
	}

	for(int i = 0; i < TopSPTree.size(); i++)
	{
		if(TopSPTree[i]->PI < seq->PI)
		{
			TopSPTree.insert(TopSPTree.begin() + i, seq);
			return;
		}
	}

	TopSPTree.push_back(seq);
}

vector<Sequence> InsertIntoResSet(Sequence seq, vector<Sequence> seqVec)
{
	if(seqVec.empty() == true)
	{
		seqVec.push_back(seq);
		return seqVec;
	}

	for(int i = 0; i < seqVec.size(); i++)
	{
		if(seqVec[i].PI < seq.PI)
		{
			seqVec.insert(seqVec.begin() + i, seq);
			return seqVec;
		}
	}

	seqVec.push_back(seq);
	return seqVec;
}

vector<Sequence*> InsertIntoResSet(Sequence* seq, vector<Sequence*> seqVec)
{
	if(seqVec.empty() == true)
	{
		seqVec.push_back(seq);
		return seqVec;
	}

	for(int i = 0; i < seqVec.size(); i++)
	{
		if(seqVec[i]->PI < seq->PI)
		{
			seqVec.insert(seqVec.begin() + i, seq);
			return seqVec;
		}
	}

	seqVec.push_back(seq);
	return seqVec;
}


///////////////////////////////////////////////////////////////

vector<Sequence> VerifyCandidatesN(vector<Sequence> candidates)
{
	vector<Sequence> resultingSeq;

	for(int i = 0; i < candidates.size(); i++)
	{
		double index = 1.5;

		for(int j = candidates[i].tailEventSet.size() - 1; j >= 0; j--)
		{

			//double min = candidates[i].PI;
			double pr = CalculatePI(candidates[i].tailEventSet[j]);
			//cout << candidates[i].sequence[j] << " " << pr << endl;

			if(pr < index)
			{
				index = pr;
				candidates[i].PI = index;
			}
		}
		//if(index > 0)
		{
		//	cout << index << endl;
		}
		//cout << index << endl;

		/*if(index > theta)
		{
			candidates[i]->PI = index;
			resultingSeq.push_back(candidates[i]);
		}*/


		if(index > theta)
		{

			resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
							/*

			if(Top.size() < (N - 1) || Top.empty() == true)
			{
				candidates[i].PI = index;
				resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
				InsertIntoSequSet(candidates[i]);
			}
			else if(Top.size() == (N - 1))
			{
				candidates[i].PI = index;
				resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
				InsertIntoSequSet(candidates[i]);
				theta = Top[N - 1].PI;
			}
			else
			{
				candidates[i].PI = index;
				resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
				InsertIntoSequSet(candidates[i]);
				theta = Top[N - 1].PI;

				//if(index > theta)
				{


					//delete all below theta
					//for(int q = 0; q < Top.size(); q++)
					{
						//if(Top[q].PI < theta)
						{
							Top.erase(Top.begin() + N);
							//cout << theta << " " << Top[q].PI << " " <<  <<" " << Top.size() << endl;
						}
					}

					//for(int w = 0; w < resultingSeq.size(); w++)
					{
						//if(resultingSeq[w].PI < theta)
						{
							resultingSeq.erase(resultingSeq.begin() + (resultingSeq.size() - 1));
						}
					}

				}
			}*/
		}
	}

	for(int i = 0; i < candidates.size(); i++)
	{
		//delete candidates[i];
	}

	return resultingSeq;
}

vector<Sequence> CandidateGen(vector<Sequence> LengthMSeq)
		{
	vector<Sequence> candidates;

	for(int i = 0; i < LengthMSeq.size(); i++)
	{
		for(int j = 0; j < LengthMSeq.size(); j++)
		{
			bool toConnect = true;

			if(i != j)
			{

				for(int l = 1; l < LengthMSeq[i].sequence.size(); l++)
				{
					if(LengthMSeq[i].sequence[l] != LengthMSeq[j].sequence[l-1])
					{
						toConnect = false;
					}
				}

				if(toConnect == true)
				{

					Sequence seq;
					GseqID++;
					seq.seqID = GseqID;

					for(int l = 0; l < LengthMSeq[i].sequence.size(); l++)
					{
						seq.sequence.push_back(LengthMSeq[i].sequence[l]);
						//cout << seq.sequence[l] <<  " ";
					}
					seq.sequence.push_back(LengthMSeq[j].sequence[LengthMSeq[j].sequence.size() - 1]);

					//if(isTheSame(seq) == false)
					{
					//cout << seq.sequence[seq.sequence.size() - 1] <<  " " << endl;

					for(int l = 0; l < LengthMSeq[i].tailEventSet.size(); l++)
					{
						//seq.tailEventSet.push_back(LengthMSeq[i].tailEventSet[l]);
					}


					seq.tailEventSet.push_back(ForwardSweep(LengthMSeq[i].tailEventSet[LengthMSeq[i].tailEventSet.size() - 1],
							LengthMSeq[j].tailEventSet[LengthMSeq[j].tailEventSet.size() - 1]));

					candidates.push_back(seq);
					}
				}
			}
		}
	}

	return candidates;
		}
/*
void Miner()
{
	vector<Sequence> Length2Seq;

	for(int i = 0; i < sortedDataset.size(); i++) //create 1-length sequences
	{
		for(int j = 0; j < sortedDataset.size(); j++)
		{
			if(i != j)
			{
				Sequence seq;
				//cout << sortedDataset[i][0].eventType << endl;
				seq.sequence.push_back(sortedDataset[i][0].eventType);
				seq.sequence.push_back(sortedDataset[j][0].eventType);

				seq.tailEventSet.push_back(sortedDataset[i]);
				//cout << i << " " << j << endl;
				vector<STPoint> I2 = ForwardSweep(seq.tailEventSet[0], sortedDataset[j]);

				if(I2.empty() == false)
				{
					seq.tailEventSet.push_back(I2);
					seq.PI = CalculatePI(seq.tailEventSet[1]);
					//if(seq.PI >= theta)
					{

						GseqID++;
						seq.seqID = GseqID;
						Length2Seq.push_back(seq);
					}
				}
			}
		}
	}

	Length2Seq = VerifyCandidatesN(Length2Seq);

	SequencesSet.push_back(Length2Seq);


	int k = 0;
	while(SequencesSet[k].empty() == false)
	{

		vector<Sequence> candidates = CandidateGen(SequencesSet[k]);

		if(candidates.empty() == true)
		{
			break;
		}

		vector<Sequence> LengthKSeq = VerifyCandidatesN(candidates);

		SequencesSet.push_back(LengthKSeq);


		k++;

	}
}
*/
//////////////

vector<Sequence*> VerifyCandidatesSPTree(vector<Sequence*> candidates)
{
	vector<Sequence*> resultingSeq;

	for(int i = 0; i < candidates.size(); i++)
	{
		double index = 1.5;

		//for(int j = candidates[i]->sequence.size() - 1; j >= candidates[i]->sequence.size() - 1; j--)
		{

			index = candidates[i]->PI;
			double pr = CalculatePI(candidates[i]->tailEventSet[0]);

			if(pr < index)
			{
				index = pr;
			}
		}

		//cout << "PI ca;cu;ated" << endl;
		//if(index > 0)
		{
			//cout << index << endl;
		}
		//cout << index << endl;

		if(index > theta)
		{

			candidates[i]->PI = index;
			resultingSeq.push_back(candidates[i]);

		}
		else
		{
			for(int w = 0; w < candidates[i]->firstParent->children.size(); w++)
			{
					if(candidates[i]->firstParent->children[w]->sequence[0] == candidates[i]->sequence[0])
					{
						candidates[i]->firstParent->children.erase(candidates[i]->firstParent->children.begin() + w);
						delete candidates[i];
					}
			}
		}
		/*
		if(index > theta)
				{
			//resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
			//InsertIntoSequSet(candidates[i]);

					if(TopSPTree.size() < (N - 1) || TopSPTree.empty() == true)
					{
						candidates[i]->PI = index;
						resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
						InsertIntoSequSet(candidates[i]);
					}
					else if(TopSPTree.size() == (N - 1))
					{
						candidates[i]->PI = index;
						resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
						InsertIntoSequSet(candidates[i]);
						theta = TopSPTree[N - 1]->PI;
					}
					else
					{
						candidates[i]->PI = index;
						resultingSeq = InsertIntoResSet(candidates[i], resultingSeq);
						InsertIntoSequSet(candidates[i]);
						theta = TopSPTree[N - 1]->PI;

						//if(index > theta)
						{


							//delete all below theta
							//for(int q = 0; q < TopSPTree.size() - 1; q++)
							{
								//if(TopSPTree[q]->PI < theta)
								{
									TopSPTree.erase(TopSPTree.begin() + N);
									//cout << TopSPTree.size() << endl;
								}
							}

							//for(int w = 0; w < resultingSeq.size(); w++)
							{
								//if(resultingSeq[w]->PI < theta)
								{

									for(int w = 0; w < resultingSeq[resultingSeq.size() - 1]->firstParent->children.size(); w++)
									{

										if(resultingSeq[resultingSeq.size() - 1]->firstParent->children[w]->sequence[0] == resultingSeq[resultingSeq.size() - 1]->sequence[0])
										{
											//cout <<resultingSeq[resultingSeq.size() - 1]->firstParent->children.size() << endl;
											resultingSeq[resultingSeq.size() - 1]->firstParent->children.erase(resultingSeq[resultingSeq.size() - 1]->firstParent->children.begin() + w);
											//cout <<resultingSeq[resultingSeq.size() - 1]->firstParent->children.size() << endl;
										}
									}
									//delete resultingSeq[(resultingSeq.size() - 1)];
									resultingSeq.erase(resultingSeq.begin() + (resultingSeq.size() - 1));
								}
							}

						}
					}
				}
		else
		{
			for(int w = 0; w < candidates[i]->firstParent->children.size(); w++)
			{
				if(candidates[i]->firstParent->children[w]->sequence[0] == candidates[i]->sequence[0])
				{
					candidates[i]->firstParent->children.erase(candidates[i]->firstParent->children.begin() + w);
				}
			}
		}
	}
	*/

	}
	return resultingSeq;
}

void VerifyNonClosed(vector<Sequence*> LengthMSeq)
{
	for(int i = 0; i < LengthMSeq.size(); i++)
	{
		Sequence* seq1 = LengthMSeq[i];

		if(seq1->firstParent->PI == seq1->PI)
		{
			seq1->firstParent->isClosed = false;
		}
		if(seq1->secondParent->PI == seq1->PI)
		{
			seq1->secondParent->isClosed = false;
		}
	}
}


vector<Sequence*> CandidateGen(vector<Sequence*> LengthMSeq)
{
	vector<Sequence*> candidates;

	for(int i = 0; i < LengthMSeq.size(); i++)
	{
		for(int j = 0; j < LengthMSeq.size(); j++)
		{

			bool toConnect = true;
			//cout << "p1" << endl;
			if(i != j)
			{

			Sequence* seq1 = LengthMSeq[i];
			Sequence* seq2 = LengthMSeq[j]->firstParent;

			while(seq2->firstParent != NULL)
			{
				//cout << seq2 << endl;
				if(seq1->sequence[0] != seq2->sequence[0])
				{
					//cout << "false" << endl;
					toConnect = false;
					break;
				}
				if(seq1->firstParent != NULL)
				{
					seq1 = seq1->firstParent;
				}
				if(seq2->firstParent != NULL)
				{
					seq2 = seq2->firstParent;
				}
			}

			//cout << "match" << endl;

			if(toConnect == true){
				//cout << "toConnect" << endl;
			//cout << LengthMSeq[i].secondParent->children.size() << endl;
			Sequence* seq = new Sequence;
			GseqID++;
			seq->seqID = GseqID;

			seq->sequence.push_back(LengthMSeq[j]->sequence[0]);

			//cout << LengthMSeq[i].secondParent->children[j]->sequence[LengthMSeq[i].secondParent->children[j]->sequence.size() - 1] << endl;
			//for(int l = 0; l < seq->sequence.size() - 1; l++)
			{
				//seq->tailEventSet.push_back(LengthMSeq[i]->tailEventSet[l]);
				//cout << seq->sequence[l] <<  " ";
			}

			//cout << seq->sequence[seq->sequence.size() - 1] <<  " " << endl;
			seq->tailEventSet.push_back(ForwardSweep(LengthMSeq[i]->tailEventSet[0],
					LengthMSeq[j]->tailEventSet[0]));

			seq->firstParent = LengthMSeq[i];
			seq->secondParent = LengthMSeq[j];

			//cout << "Cangen" << endl;
			candidates.push_back(seq);
			}
			}
		}
	}
	return candidates;
}

bool isUnique(Sequence* firstSeq, Sequence* secondSeq)
{
	bool toConnect = true;

	Sequence* seqScheck = firstSeq;


	while(seqScheck != NULL)
	{

		if(seqScheck->sequence[0] == secondSeq->sequence[0])
		{

			toConnect = false;
			break;
		}
		seqScheck = seqScheck->firstParent;
	}

	return toConnect;
}
vector<Sequence*> CandidateGenSPTree(vector<Sequence*> LengthMSeq)
{
	vector<Sequence*> candidates;

	for(int i = 0; i < LengthMSeq.size(); i++)
	{
		for(int j = 0; j < LengthMSeq[i]->secondParent->children.size(); j++)
		{
			//if(isUnique(LengthMSeq[i], LengthMSeq[i]->secondParent->children[j]) == true)
			{
			//cout << LengthMSeq[i]->secondParent->children.size() << endl;
			Sequence* seq = new Sequence;
			GseqID++;
			seq->seqID = GseqID;

			for(int l = 0; l < LengthMSeq[i]->sequence.size(); l++)
			{
				//seq->sequence.push_back(LengthMSeq[i]->sequence[l]);
			}
			seq->sequence.push_back(LengthMSeq[i]->secondParent->children[j]->sequence[0]);

			//cout << LengthMSeq[i].secondParent->children[j]->sequence[LengthMSeq[i].secondParent->children[j]->sequence.size() - 1] << endl;
			//for(int l = 0; l < seq->sequence.size() - 1; l++)
			{
				//seq->tailEventSet.push_back(LengthMSeq[i]->tailEventSet[l]);
				//cout << seq->sequence[l] <<  " ";
			}

			//cout << seq->sequence[seq->sequence.size() - 1] <<  " " << endl;
			seq->tailEventSet.push_back(ForwardSweep(LengthMSeq[i]->tailEventSet[0],
					LengthMSeq[i]->secondParent->children[j]->tailEventSet[0]));


			seq->firstParent = LengthMSeq[i];
			seq->secondParent = (LengthMSeq[i]->secondParent->children[j]);
			seq->PI = seq->firstParent->PI;

			seq->firstParent->children.push_back(seq);
			//cout << "Cangen" << endl;
			candidates.push_back(seq);
			}
		}
	}
	return candidates;
}
int CountPatterns()
{
	int count = 0;
	for(int i = 1; i < SequencesSetSPTree.size(); i++ )
	{
		for(int j = 0; j < SequencesSetSPTree[i].size(); j++)
		{
			if (SequencesSetSPTree[i][j]->isClosed == true )
			{
				count++;
			}
		}
	}

	return count;
}

void MinerSPTree()
{
	vector<Sequence*> Length2Seq;
	vector<Sequence*> Length1Seq;

	for(int i = 0; i < sortedDataset.size(); i++) //create 1-length sequences
	{
		Sequence *seq = new Sequence;
		seq->sequence.push_back(sortedDataset[i][0].eventType);

		vector<STPoint *> pointersData;

				for(int j = 0; j < sortedDataset[i].size(); j++)
				{
					pointersData.push_back(&sortedDataset[i][j]);
				}

				seq->tailEventSet.push_back(pointersData);

		GseqID++;
		seq->seqID = GseqID;

		Length1Seq.push_back(seq);
		//cout << "Processed rvrnt type" << i << endl;
	}
	SequencesSetSPTree.push_back(Length1Seq);

	for(int i = 0; i < Length1Seq.size(); i++) //create 1-length sequences
	{
		for(int j = 0; j < Length1Seq.size(); j++)
		{
			//if(i != j)
			{
				//cout << "Processed " << i << " " << j << " " << Length1Seq.size() << endl;
				Sequence* seq = new Sequence;

				//seq->sequence.push_back(Length1Seq[i]->sequence[0]);
				seq->sequence.push_back(Length1Seq[j]->sequence[0]);

				//seq->tailEventSet.push_back(Length1Seq[i]->tailEventSet[0]);
				vector<STPoint *> I2 = ForwardSweep(Length1Seq[i]->tailEventSet[0], Length1Seq[j]->tailEventSet[0]);

				//cout << "err" << endl;
				if(I2.empty() == false)
				{
					seq->tailEventSet.push_back(I2);
					seq->PI = CalculatePI(seq->tailEventSet[0]);
					//if(seq->PI >= theta)
					{

						GseqID++;
						seq->seqID = GseqID;
						seq->firstParent = Length1Seq[i];
						seq->secondParent = Length1Seq[j];

						seq->firstParent->children.push_back(seq);

						Length2Seq.push_back(seq);

					}

					//cout << i << " " << j << endl;
				}
			}
		}
	}
	cout << "Phase 1" << endl;
	Length2Seq = VerifyCandidatesSPTree(Length2Seq);

	SequencesSetSPTree.push_back(Length2Seq);

	int k = 1;
	while(SequencesSetSPTree[k].empty() == false)
	{

		vector<Sequence*> candidates = CandidateGenSPTree(SequencesSetSPTree[k]);

		if(candidates.empty() == true)
		{
			break;
		}

		vector<Sequence*> LengthKSeq = VerifyCandidatesSPTree(candidates);
		//cout << "v" << k << " " <<LengthKSeq.size() << endl;
		SequencesSetSPTree.push_back(LengthKSeq);

		k++;
	}

	cout << "Phase 2" << endl;

}

void MinerSPTreeClosed()
{
	vector<Sequence*> Length2Seq;
	vector<Sequence*> Length1Seq;

	for(int i = 0; i < sortedDataset.size(); i++) //create 1-length sequences
	{
		Sequence *seq = new Sequence;
		seq->sequence.push_back(sortedDataset[i][0].eventType);

		vector<STPoint *> pointersData;

		for(int j = 0; j < sortedDataset[i].size(); j++)
		{
			pointersData.push_back(&sortedDataset[i][j]);
		}

		seq->tailEventSet.push_back(pointersData);
		GseqID++;
		seq->seqID = GseqID;

		Length1Seq.push_back(seq);
		//cout << "Processed rvrnt type" << i << endl;
	}

	SequencesSetSPTree.push_back(Length1Seq);

	for(int i = 0; i < Length1Seq.size(); i++) //create 1-length sequences
	{
		for(int j = 0; j < Length1Seq.size(); j++)
		{
			//if(i != j)
			{
				//cout << "Processed " << i << " " << j << " " << Length1Seq.size() << endl;
				Sequence* seq = new Sequence;

				//seq->sequence.push_back(Length1Seq[i]->sequence[0]);
				seq->sequence.push_back(Length1Seq[j]->sequence[0]);

				//seq->tailEventSet.push_back(Length1Seq[i]->tailEventSet[0]);
				vector<STPoint *> I2 = ForwardSweep(Length1Seq[i]->tailEventSet[0], Length1Seq[j]->tailEventSet[0]);
				//cout << "err" << endl;
				if(I2.empty() == false)
				{
					seq->tailEventSet.push_back(I2);
					seq->PI = CalculatePI(seq->tailEventSet[0]);
					//if(seq->PI >= theta)
					{

						GseqID++;
						seq->seqID = GseqID;
						seq->firstParent = Length1Seq[i];
						seq->secondParent = Length1Seq[j];

						seq->firstParent->children.push_back(seq);

						Length2Seq.push_back(seq);

					}

					//cout << i << " " << j << endl;
				}
			}
		}
	}
	cout << "Phase 1" << endl;
	Length2Seq = VerifyCandidatesSPTree(Length2Seq);

	SequencesSetSPTree.push_back(Length2Seq);

	int k = 1;
	while(SequencesSetSPTree[k].empty() == false)
	{

		vector<Sequence*> candidates = CandidateGenSPTree(SequencesSetSPTree[k]);

		if(candidates.empty() == true)
		{
			break;
		}

		vector<Sequence*> LengthKSeq = VerifyCandidatesSPTree(candidates);
		//cout << "v" << k << " " <<LengthKSeq.size() << endl;
		SequencesSetSPTree.push_back(LengthKSeq);
		VerifyNonClosed(SequencesSetSPTree[SequencesSetSPTree.size() - 1]);

		k++;
	}

	cout << "Phase 2" << endl;

}


//##################################################################

void LoadDataset(string Path) //za³aduj zbiór danych
{
	size = CountInstances(Path); // zlicz l. instancji w pliku
	data = new STPoint[size]; // rozszerz data

	uchwyt.open(Path);
	for(int i = 0; i < size; i++)
	{
		string line;
		getline(uchwyt, line);
		stringstream linestream(line);
		string dataPortion;

		if(line != ""){
			getline(linestream, dataPortion, ' ');
			data[i].eventID = stoi(dataPortion);
			getline(linestream, dataPortion, ' ');
			data[i].eventType = dataPortion;
			getline(linestream, dataPortion, ' ');
			data[i].spatialX = stod(dataPortion);
			getline(linestream, dataPortion, ' ');
			data[i].spatialY = stod(dataPortion);
			getline(linestream, dataPortion, ' ');
			data[i].temporal = stod(dataPortion);
		}

		//cout << i << endl;
	}
	uchwyt.close();
}

void TransformData()
{
	for(int i = 0; i < size; i++)
	{
		InsertInstance(data[i]);
	}
}

void InsertInstance(STPoint instance)
{
	for(int i = 0; i < dataset.size(); i++)
	{
		if(dataset[i][0].eventType == instance.eventType)
		{
			dataset[i].push_back(instance);
			return;
		}
	}

	vector<STPoint> vect;
	vect.push_back(instance);
	dataset.push_back(vect);
}

void PrintDataset()
{

	for(int i = 0; i < dataset.size(); i++)
	{
		for(int j = 0; j < dataset[i].size(); j++)
		{
			cout << dataset[i][j].eventID << ' ' << dataset[i][j].eventType << '\t';
			//cout << i;
		}

		cout << endl;
	}
}

void PrintSortedDataset()
{

	for(int i = 0; i < dataset.size(); i++)
	{
		for(int j = 0; j < dataset[i].size(); j++)
		{
			cout << sortedDataset[i][j].temporal << ' ' << sortedDataset[i][j].eventType << '\t';
			//cout << i;
		}

		cout << endl;
	}

}

void PrintSequencesTop()
{
	fstream results;
	results.open("ResultsTop.txt", fstream::out);

	for(int i = Top.size() - 1; i >= 0;  i--)
	{
		for(int j = Top[i].sequence.size() - 1; j >= 0; j--)
		{
			cout << i << Top[i].sequence[j] << " ";
		}
		cout << endl;
	}
	results.close();
}

void PrintSequencesTopSPTree()
{
	fstream results;
	results.open("ResultsTop.txt", fstream::out);

	for(int i = 0; i < TopSPTree.size();  i++)
	{

		Sequence *seq = TopSPTree[i];

		cout << "i " << seq->PI << " ";
		results <<  "i " << seq->PI << " ";
		while(seq != NULL)
		{
			cout << seq->sequence[0] << " <- " ;
			results << seq->sequence[0] << " <- ";
			seq = seq->firstParent;
		}

		cout << endl;
		results << endl;
	}
	results.close();
}



void PrintSequences()
{
	fstream results;
	results.open("ResultsSequencesClosed3_closedNie.txt", fstream::out);

	for(int i = SequencesSetSPTree.size() - 1; i >= 0;  i--)
	{
		for(int j = 0; j < SequencesSetSPTree[i].size();  j++)
		{
			Sequence *seq = SequencesSetSPTree[i][j];

			if(seq->isClosed == true)
			{
				int count = 0;
				double aPI = seq->PI;
				cout << "i " << seq->PI << " ";
				results <<  "i " << seq->PI << " ";

				while(seq != NULL)
				{
					if(seq->PI == aPI)
					{
						count++;
					}
					cout << seq->sequence[0] << " <- " ;
					results << seq->sequence[0] << " <- ";
					seq = seq->firstParent;
				}

				cout << "Closed" << endl;
				results << "Closed" << endl;
			}
		}
	}
	results.close();
}

void ClearSequencesSet()
{
	SequencesSet.clear();
}

void ClearDataset()
{
	dataset.clear();
}

void ClearSortedDataset()
{
	sortedDataset.clear();
}

void ClearStructures()
{
	dataset.clear();
	sortedDataset.clear();

	SequencesSet.clear();
	Top.clear();
	for(int i = 0; i < SequencesSetSPTree.size(); i++)
	{
		for(int j = 0; j < SequencesSetSPTree[i].size(); j++)
		{
			delete SequencesSetSPTree[i][j];
		}
	}
	SequencesSetSPTree.clear();
	TopSPTree.clear();
}

int CountInstances(string path) // zlicz liczbê instancji w pliku
{
	uchwyt.open(path);
	string line;

	int numInstances = 0;

	while(uchwyt.eof() != true)
	{
		getline(uchwyt, line);

		if(line != "")
		{
			numInstances++;
		}
	}

	uchwyt.close();
	return numInstances;
}
