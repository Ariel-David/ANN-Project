/***********************************************************************
*                       Neural Networks 							   *
*               Project number 2 - Back Propagation					   *
*                   												   *
*                     This project executed by:						   *
*                         Volinsky Irina							   *
*                         ID:  310598255							   *
************************************************************************
*/

//This is a Back Propagation network. It consists layers:
//25 neurons on input layer, 10 neurons on hidden layer and one
//neuron on output layer.

//This programm use function: F(NET) = tanh (NET), that takes values
//from -1 to +1.
//The values of the neurons in the hidden layer are continuous.
//The values of the input and output neurons are diskreet.

//This program do not print anything to display, and all results of
//this programm will be in file "result.txt" and "bias_result.txt" 
//after runing of this programm.
//Before runing programm againe, please delete old file with results.

using namespace std;
#define _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE 
#include <iostream>
#include<stdlib.h>
#include<time.h>
#include<string>
#include<math.h>
#include<fcntl.h>
#include<sys\stat.h>
#include<io.h>
#include "Pattrens.dat"      //File with patterns for input end output.

#define Low           -1
#define Hi	          +1
#define Bias          1
#define InputNeurons  100	  
#define HiddenNeurons 50

typedef int InArr[InputNeurons];

typedef enum boolean boolean;

class Data
{
public:
	InArr* Input;
	float* Output;
	int Units;     //Numbers (units) in input ( and output ) now.
	int order_arr_Test[9];
	int order_arr15[15];
	int order_arr30[30];
	int order_arr57[57];

	Data();
	~Data();

	void SetUnorderedNumbers_Test(int);
	void SetUnorderedNumbers15(int);
	void SetUnorderedNumbers30(int);
	void SetUnorderedNumbers57(int);

	//Set input and output vectors from patterns.
	bool SetInputOutput(char[][Y][X], double*, int);

	//Free memory of Input and Output units
	void Reset();

	bool SetInputOutputRand_Test(char[][Y][X], double*, int);
	bool SetInputOutputRand15(char[][Y][X], double*, int);
	bool SetInputOutputRand30(char[][Y][X], double*, int);
	bool SetInputOutputRand57(char[][Y][X], double*, int);
};

class BackPropagationNet
{
private:
	//Input to network (+ bias = 1 if it is).
	int    InputLayer[InputNeurons + 1];

	//Output from hidden layer -> it is input to output layer.
	float  HiddenLayer[HiddenNeurons + 1];

	//Output of network - one neuron.
	float  OutputLayer;	                    //Takes values: -1 or +1.

	float  WeigthsOut[HiddenNeurons + 1];
	float  WeigthsHidd[HiddenNeurons + 1][InputNeurons + 1];
	float  nu;                                       //Learning rate.
	float  Threshold1;
	float  Threshold2;
	//It was error now ?. If error occured, then NetError = true, 
	//else NetError = false.
	bool  NetError;

	float RandomEqualReal(float, float);

	float Sig(float);

	float derivative(float);

	//Calculate output for current input (without Bias).
	void CalculateOutput();

	void ItIsError(float);           //NetError = true if it was error.

	void AdjustWeigths(int);				          

public:
	//Initialization of weigths and variables.
	BackPropagationNet();

	//Initialize all and randomly weigths.
	void  Initialize();


	//Train network up to 90% success or up to 20000 cycles 
	//(without Bias).
	bool TrainNet(Data&);

	//Testing of network (without Bias). Return success percent.
	int TestNet(Data&);
	bool TrainNetRand(Data&);

	const int ReturnOutput() { return OutputLayer; };
	
	float LearningRate() { return nu; };
	float Threshold1Value() { return Threshold1; };
	float Threshold2Value() { return Threshold2; };

};

//********************* CLASS BACKPROPAGATIONNET *******************

BackPropagationNet::BackPropagationNet()
{
	nu = 0.1f;

	srand((unsigned)time(NULL));
	Initialize();
}


//_________________________________________________________________________


void BackPropagationNet::Initialize()
{
	int i, j;
	Threshold1 = 0.3f;
	Threshold2 =  0.6f;
	NetError = false;
	

	//Randomize weigths (initialize).
	for (i = 0; i < HiddenNeurons + 1; i++)
		WeigthsOut[i] = RandomEqualReal(-1.0f, 1.0f);

	for (i = 0; i < HiddenNeurons + 1; i++)
	{
		for (j = 0; j < InputNeurons + 1; j++)
			WeigthsHidd[i][j] = RandomEqualReal(-1.0f, 1.0f);
	}
}
//_________________________________________________________________________


//Return randomaly numbers from LowN to HighN
float BackPropagationNet::RandomEqualReal(float LowN, float HighN)
{
	return ((float)rand() / RAND_MAX) * (HighN - LowN) + LowN;
}


//_________________________________________________________________________

float BackPropagationNet::Sig(float x) {
	float ans;
	ans = 1/(1+exp(-1*x));
	return ans;
}

float BackPropagationNet::derivative(float x)
{
	float ans;
	ans = (exp(-1 * x)) / ((1 + exp(-1 * x)) * (1 + exp(-1 * x)));
	return ans;
}
//_________________________________________________________________________


void BackPropagationNet::CalculateOutput()
{
	float Sum;

	//Calculate output for hidden layer.
	for (int i = 0; i < HiddenNeurons; i++)
	{
		Sum = 0.0f;
		for (int j = 0; j < InputNeurons; j++)
		{
			Sum += WeigthsHidd[i][j] * InputLayer[j];
		}

		HiddenLayer[i] = Sig(Sum);
	}

	//Calculate output for output layer.
	Sum = 0.0f;

	for (int n = 0; n < HiddenNeurons; n++)
		Sum += WeigthsOut[n] * HiddenLayer[n];

	//Make decision about output neuron.
	float temp = Sig(Sum);
	if (Sig(Sum) < 0.333333)//<0.33 circle
		OutputLayer = 0;

	else if (Sig(Sum) < 0.666666 && Sig(Sum) >= 0.333333)//[0.33,0.66]
	OutputLayer = 0.5; // elips

	else						                     //We can not decide.
		OutputLayer = 1; // triangle
}


//_________________________________________________________________________


void BackPropagationNet::ItIsError(float Target)
{
	if (((float)Target - OutputLayer))
		NetError = true;
	else
		NetError = false;
}


//_________________________________________________________________________



void BackPropagationNet::AdjustWeigths(int Target)
{
	int i, j;
	float hidd_deltas[HiddenNeurons], out_delta;

	//Calcilate deltas for all layers.

	out_delta = derivative(OutputLayer) * (Target - OutputLayer);

	for (i = 0; i < HiddenNeurons; i++)
		hidd_deltas[i] = derivative(HiddenLayer[i]) * out_delta * WeigthsOut[i];

	//Change weigths.
	for (i = 0; i < HiddenNeurons; i++)
		WeigthsOut[i] = WeigthsOut[i] + (nu * out_delta * HiddenLayer[i]);

	for (i = 0; i < HiddenNeurons; i++)
	{
		for (j = 0; j < InputNeurons + 1; j++)
			WeigthsHidd[i][j] = WeigthsHidd[i][j] + (nu * hidd_deltas[i] * InputLayer[j]);
	}
}


//_________________________________________________________________________


bool BackPropagationNet::TrainNet(Data& data_obj)
{
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK: SEQUENTIALY" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;

	do
	{
		Error = 0;
		loop++;

		cout << "Thresholds =    " << Threshold1 <<" , "<< Threshold2 << endl;

		//Printing the number of loop.
		if (loop < 10)
			cout << "Training loop:  " << loop << "       ...   ";
		if (loop >= 10 && loop < 100)
			cout << "Training loop:  " << loop << "      ...   ";
		if (loop >= 100 && loop < 1000)
			cout << "Training loop:  " << loop << "     ...   ";
		if (loop >= 1000 && loop < 10000)
			cout << "Training loop:  " << loop << "    ...   ";
		else if (loop >= 10000)
			cout << "Training loop:  " << loop << "   ...   ";

		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < InputNeurons; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput();
			ItIsError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths(data_obj.Output[i]);
			}
		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;

	

	} while (Success < 85 && loop <= 20000);

	if (loop > 20000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}


//_________________________________________________________________________


int BackPropagationNet::TestNet(Data& data_obj)
{
	int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
	cout << "                    TEST NETWORK SEQUENTIALY " << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;

	cout << "Test network    ...  ";

	//Train network (do one cycle).
	for (int i = 0; i < data_obj.Units; i++)
	{
		//Set current input.
		for (j = 0; j < InputNeurons; j++)
			InputLayer[j] = data_obj.Input[i][j];

		CalculateOutput();
		ItIsError(data_obj.Output[i]);

		//Error = sum of errors in this one cycle of test.
		if (NetError)
			Error++;
	}

	Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
	cout << Success << "%   success" << endl;

	return Success;
}
//_________________________________________________________________________

bool BackPropagationNet::TrainNetRand(Data& data_obj) {
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK RANDOMALY" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;

	do
	{
		Error = 0;
		loop++;

		cout << "Thresholds =    " << Threshold1 << " , " << Threshold2 << endl;

		//Printing the number of loop.
		if (loop < 10)
			cout << "Training loop:  " << loop << "       ...   ";
		if (loop >= 10 && loop < 100)
			cout << "Training loop:  " << loop << "      ...   ";
		if (loop >= 100 && loop < 1000)
			cout << "Training loop:  " << loop << "     ...   ";
		if (loop >= 1000 && loop < 10000)
			cout << "Training loop:  " << loop << "    ...   ";
		else if (loop >= 10000)
			cout << "Training loop:  " << loop << "   ...   ";

		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < InputNeurons; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput();
			ItIsError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths(data_obj.Output[i]);
			}
		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;



	} while (Success < 80 && loop <= 20000);

	if (loop > 20000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}



//************************ CLASS DATA *************************************


Data::Data()
{
	Units = 0;
}


//_________________________________________________________________________


Data::~Data()
{
	Reset();
}


//_________________________________________________________________________


void Data::Reset()
{
	Units = 0;
	delete[] Input;
	delete[] Output;
}


//_________________________________________________________________________

bool Data::SetInputOutputRand_Test(char In[][Y][X], double* Out, int num_patterns) {
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new float[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	SetUnorderedNumbers_Test(num_patterns);

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr_Test[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		Output[i] = Out[order_arr_Test[i]];
	}

	return true;
}


//_________________________________________________________________________
bool Data::SetInputOutputRand15(char In[][Y][X], double* Out, int num_patterns) {
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new float[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	SetUnorderedNumbers15(num_patterns);

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr15[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		Output[i] = Out[order_arr15[i]];
	}

	return true;
}

//_________________________________________________________________________

bool Data::SetInputOutputRand30(char In[][Y][X], double* Out, int num_patterns) {
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new float[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	SetUnorderedNumbers30(num_patterns);

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr30[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		Output[i] = Out[order_arr30[i]];
	}

	return true;
}

//_________________________________________________________________________

bool Data::SetInputOutputRand57(char In[][Y][X], double* Out, int num_patterns) {
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new float[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	SetUnorderedNumbers57(num_patterns);

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr57[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		Output[i] = Out[order_arr57[i]];
	}

	return true;
}
//_________________________________________________________________________

void Data::SetUnorderedNumbers_Test(int size)
{
	int number, index;

	for (int i = 0; i < size; i++)                      //Initialize array.
		order_arr_Test[i] = -1;

	for (number = 0; number < size; number++)
	{
		index = rand() % size;
		if (order_arr_Test[index] == -1)         //If the place is empty.
		{
			order_arr_Test[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr_Test[index] != -1)
			{
				index++;
				index = index % size;
			}

			order_arr_Test[index] = number;       //We finded empty place.
		}
	}
}

//_________________________________________________________________________

void Data::SetUnorderedNumbers15(int size)
{
	int number, index;

	for (int i = 0; i < size; i++)                      //Initialize array.
		order_arr15[i] = -1;

	for (number = 0; number < size; number++)
	{
		index = rand() % size;
		if (order_arr15[index] == -1)         //If the place is empty.
		{
			order_arr15[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr15[index] != -1)
			{
				index++;
				index = index % size;
			}

			order_arr15[index] = number;       //We finded empty place.
		}
	}
}


//_________________________________________________________________________

void Data::SetUnorderedNumbers30(int size)
{
	int number, index;

	for (int i = 0; i < size; i++)                      //Initialize array.
		order_arr30[i] = -1;

	for (number = 0; number < size; number++)
	{
		index = rand() % size;
		if (order_arr30[index] == -1)         //If the place is empty.
		{
			order_arr30[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr30[index] != -1)
			{
				index++;
				index = index % size;
			}

			order_arr30[index] = number;       //We finded empty place.
		}
	}
}

//_________________________________________________________________________

void Data::SetUnorderedNumbers57(int size)
{
	int number, index;

	for (int i = 0; i < size; i++)                      //Initialize array.
		order_arr57[i] = -1;

	for (number = 0; number < size; number++)
	{
		index = rand() % size;
		if (order_arr57[index] == -1)         //If the place is empty.
		{
			order_arr57[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr57[index] != -1)
			{
				index++;
				index = index % size;
			}

			order_arr57[index] = number;       //We finded empty place.
		}
	}
}

//_________________________________________________________________________


bool Data::SetInputOutput(char In[][Y][X], double* Out, int num_patterns)
{
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new float[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[n][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		Output[i] = Out[i];
	}

	return true;
}


//***************************** MAIN **************************************


void main()
{

	Data data_obj;
	BackPropagationNet back_prop_obj;
	bool flag;

	cout << "Back Propagation Network" << endl << endl;
	cout << "This programm will not print anything onto display." << endl;
	cout << "All results of this programm will be in files: result_5_GROUPS.txt,result_10_GROUPS.txt,result_19_GROUPS.txt " << endl;
	cout << "after this programm will stop to run." << endl;
	cout << "Please wait until it's all done" << endl;
	cout << "Before runing this programm again, please delete the old files";

	close(1);

	//TRAINING NETWORK WITH 5 GROUPS (15 SHAPES)

	//TRAINING NETWORK: SEQUENTIALY

	int fd1 = open("result_5_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

	if (fd1 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

		if (!data_obj.SetInputOutput(TrainingInput1, TrainingOutput1, TrainPatt1))
			return;

	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd1);
		remove("result_5_GROUPS_SEQ.txt");
		fd1 = open("result_5_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

		if (fd1 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK: SEQUENTIALY

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
	close(fd1);

	//TRAINING NETWORK: RANDOMALY
	int fd2 = open("result_5_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

	if (fd2 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutputRand15(TrainingInput1, TrainingOutput1, TrainPatt1))
		return;

	while (!(flag = back_prop_obj.TrainNetRand(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd2);
		remove("result_5_GROUPS_RAND.txt");
		fd2 = open("result_5_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

		if (fd2 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK: RANDOMALY
	if (!data_obj.SetInputOutputRand_Test(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
	close(fd2);


	//TRAINING NETWORK WITH 10 GROUPS SEQ (30 SHAPES)
	int fd3 = open("result_10_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

	if (fd3 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutput(TrainingInput2, TrainingOutput2, TrainPatt2))
		return;

	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd3);
		remove("result_10_GROUPS_SEQ.txt");
		fd3 = open("result_10_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

		if (fd3 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK SEQ.

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);

	close(fd3);

	//TRAINING NETWORK WITH 10 GROUPS RANDOMALY(30 SHAPES)

	int fd4 = open("result_10_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

	if (fd4 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutputRand30(TrainingInput2, TrainingOutput2, TrainPatt2))
		return;

	while (!(flag = back_prop_obj.TrainNetRand(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd4);
		remove("result_10_GROUPS_RAND.txt");
		fd4 = open("result_10_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

		if (fd4 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK RAND.

	if (!data_obj.SetInputOutputRand_Test(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);

	close(fd4);

	//TRAINING NETWORK WITH 19 GROUPS SEQ (57 SHAPES)

	int fd5 = open("result_19_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

	if (fd5 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutput(TrainingInput3, TrainingOutput3, TrainPatt3))
		return;

	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd5);
		remove("result_19_GROUPS_SEQ.txt");
		fd5 = open("result_19_GROUPS_SEQ.txt", O_CREAT | O_RDWR, 0777);

		if (fd5 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK SEQ.

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
	close(fd5);

	//TRAINING NETWORK WITH 19 GROUPS RAND(57 SHAPES)

	int fd6 = open("result_19_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

	if (fd6 == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutputRand57(TrainingInput3, TrainingOutput3, TrainPatt3))
		return;

	while (!(flag = back_prop_obj.TrainNetRand(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd6);
		remove("result_19_GROUPS_RAND.txt");
		fd6 = open("result_19_GROUPS_RAND.txt", O_CREAT | O_RDWR, 0777);

		if (fd6 == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}
	//TEST NETWORK RANDOMALY.

	if (!data_obj.SetInputOutputRand_Test(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
	close(fd6);
}






