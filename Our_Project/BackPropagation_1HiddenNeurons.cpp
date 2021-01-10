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
#define sqr(x)        ((x)*(x))

typedef int InArr[InputNeurons];

typedef enum boolean boolean;

class Data
{
public:
	InArr* Input;
	float* Output;
	int Units;     //Numbers (units) in input ( and output ) now.

	Data();
	~Data();

	//Set input and output vectors from patterns.
	bool SetInputOutput(char[][Y][X], double*, int);

	//Free memory of Input and Output units
	void Reset();
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

	//Calculate output for current input (with Bias).
	void CalculateOutputWithBias();

	//Calculate output for current input (without Bias).
	void CalculateOutput();

	void ItIsError(float);           //NetError = true if it was error.

	void AdjustWeigthsWithBias(int);                     //With Bias.
	void AdjustWeigths(int);				          //Without Bias.

public:
	//Initialization of weigths and variables.
	BackPropagationNet();

	//Initialize all and randomly weigths.
	void  Initialize();

	//Train network up to 90% success or up to 20000 cycles 
	//(with Bias).
	bool TrainNetWithBias(Data&);

	//Train network up to 90% success or up to 20000 cycles 
	//(without Bias).
	bool TrainNet(Data&);

	//Testing of network (with Bias). Return success percent.
	int TestNetWithBias(Data&);

	//Testing of network (without Bias). Return success percent.
	int TestNet(Data&);

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


void BackPropagationNet::CalculateOutputWithBias()
{
	float Sum;

	//Calculate output for hidden layer.
	HiddenLayer[0] = (float)Bias;

	for (int i = 1; i < HiddenNeurons + 1; i++)
	{
		Sum = 0.0f;
		for (int j = 0; j < InputNeurons + 1; j++)
		{
			Sum += WeigthsHidd[i][j] * InputLayer[j];
		}
		HiddenLayer[i] = (float)(1/(1 + exp(-1 * Sum)));
	}

	//Calculate output for output layer.
	Sum = 0.0f;

	for (int n = 0; n < HiddenNeurons + 1; n++)
		Sum += WeigthsOut[n] * HiddenLayer[n];

	//Make decision about output neuron.
	if ((float)(1 / (1 + exp(Sum))) < Threshold1)
		OutputLayer = 0;
	else if ((float)(1 / (1 + exp(Sum))) < Threshold2 && ((float)(1 / (1 + exp(Sum)))) >= Threshold1)
		OutputLayer = 0.5;
	else						                     //We can not decide.
		OutputLayer = 1;
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

		HiddenLayer[i] = (float)(1/(1 + exp(-1 * Sum)));
	}

	//Calculate output for output layer.
	Sum = 0.0f;

	for (int n = 0; n < HiddenNeurons; n++)
		Sum += WeigthsOut[n] * HiddenLayer[n];

	//Make decision about output neuron.
	float temp = (float)(1/(1 + exp(-1*Sum)));
	if ((float)(1 /(1 + exp(-1 * Sum))) < Threshold1)//<0.33 circle
		OutputLayer = 0;

	else if (((float)(1/(1 + exp(-1 * Sum)))) < Threshold2 && (((float)(1/(1 + exp(-1 * Sum))))) >= Threshold1)//[0.33,0.66]
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


void BackPropagationNet::AdjustWeigthsWithBias(int Target)
{
	int   i, j;
	float hidd_deltas[HiddenNeurons + 1], out_delta;

	//Calcilate deltas for all layers.
	out_delta = (OutputLayer) * (1 - OutputLayer) * (Target - OutputLayer);

	for (i = 0; i < HiddenNeurons + 1; i++)
		hidd_deltas[i] = (HiddenLayer[i]) * (1 - HiddenLayer[i]) * out_delta * WeigthsOut[i];

	//Change weigths.
	for (i = 0; i < HiddenNeurons + 1; i++)
		WeigthsOut[i] = WeigthsOut[i] + (nu * out_delta * HiddenLayer[i]);

	for (i = 0; i < HiddenNeurons + 1; i++)
	{
		for (j = 0; j < InputNeurons + 1; j++)
			WeigthsHidd[i][j] = WeigthsHidd[i][j] +
			(nu * hidd_deltas[i] * InputLayer[j]);
	}
}


//_________________________________________________________________________


void BackPropagationNet::AdjustWeigths(int Target)
{
	int i, j;
	float hidd_deltas[HiddenNeurons], out_delta;

	//Calcilate deltas for all layers.
	out_delta = (OutputLayer) * (1 - OutputLayer) * (Target - OutputLayer);

	for (i = 0; i < HiddenNeurons; i++)
		hidd_deltas[i] = (HiddenLayer[i]) * (1 - HiddenLayer[i]) * out_delta * WeigthsOut[i];

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
bool BackPropagationNet::TrainNetWithBias(Data& data_obj)
{
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                  TRAINING NETWORK WITH BIAS" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;

	do
	{
		Error = 0;
		loop++;

		cout << "Thresholds =    " << Threshold1 << " , "<< Threshold2 << endl;

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
			InputLayer[0] = Bias;
			for (j = 0; j < InputNeurons; j++)
				InputLayer[j + 1] = data_obj.Input[i][j];

			CalculateOutputWithBias();
			ItIsError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigthsWithBias(data_obj.Output[i]);
			}
		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;

		//if (Success < 90)
			//Threshold = RandomEqualReal(0.2f, 0.9f);

	} while (Success < 90 && loop <= 20000);

	if (loop > 20000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}


//_________________________________________________________________________


bool BackPropagationNet::TrainNet(Data& data_obj)
{
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
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

		if (Success < 90) {
			Threshold1 = RandomEqualReal(0.2f, 0.9f);
			Threshold2 = RandomEqualReal(0.2f, 0.9f);
		}

	} while (Success < 90 && loop <= 20000);

	if (loop > 20000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}


//_________________________________________________________________________
int BackPropagationNet::TestNetWithBias(Data& data_obj)
{
	int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
	cout << "               TEST NETWORK WITH BIAS" << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;

	cout << "Test network    ...  ";

	//Train network (do one cycle).
	for (int i = 0; i < data_obj.Units; i++)
	{
		//Set current input.
		InputLayer[0] = Bias;
		for (j = 0; j < InputNeurons; j++)
			InputLayer[j + 1] = data_obj.Input[i][j];

		CalculateOutputWithBias();
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


int BackPropagationNet::TestNet(Data& data_obj)
{
	int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
	cout << "                    TEST NETWORK" << endl << endl;
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
	cout << "All results of this programm will be in files: result.txt ";
	cout << "and result_bias.txt  after this " << endl;
	cout << "programm will stop to run." << endl;
	cout << "Before runing this programm againe, please delete old ";
	cout << "file with results." << endl;

	close(1);
	int fd = open("result_5_GROUPS.txt", O_CREAT | O_RDWR, 0777);

	if (fd == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	//TRAINING NETWORK WITH 5 GROUPS (15 SHAPES)
	
		if (!data_obj.SetInputOutput(TrainingInput1, TrainingOutput1, TrainPatt1))
			return;

	
	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd);
		remove("result_5_GROUPS.txt");
		fd = open("result_5_GROUPS.txt", O_CREAT | O_RDWR, 0777);

		if (fd == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK 

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);

	close(fd);
	fd = open("result_10_GROUPS.txt", O_CREAT | O_RDWR, 0777);

	if (fd == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	//TRAINING NETWORK WITH 10 GROUPS (30 SHAPES)

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutput(TrainingInput2, TrainingOutput2, TrainPatt2))
		return;

	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd);
		remove("result_10_GROUPS.txt");
		fd = open("result_10_GROUPS.txt", O_CREAT | O_RDWR, 0777);

		if (fd == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK.

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);

	close(fd);
	fd = open("result_19_GROUPS.txt", O_CREAT | O_RDWR, 0777);

	if (fd == -1)
	{
		cout << "Error opening result file" << endl;
		return;
	}

	//TRAINING NETWORK WITH 19 GROUPS (57 SHAPES)

	back_prop_obj.Initialize();

	if (!data_obj.SetInputOutput(TrainingInput3, TrainingOutput3, TrainPatt3))
		return;

	while (!(flag = back_prop_obj.TrainNet(data_obj)))
	{
		back_prop_obj.Initialize();
		close(fd);
		remove("result_19_GROUPS.txt");
		fd = open("result_19_GROUPS.txt", O_CREAT | O_RDWR, 0777);

		if (fd == -1)
		{
			cout << "Error opening result file" << endl;
			return;
		}
	}

	//TEST NETWORK.

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
	close(fd);

}






