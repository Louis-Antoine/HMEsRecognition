from matplotlib import testing
from distortData import distortData
from TrainModel import trainModel
from HME_Prediction import PredictHME
from TestAccuracy import TestMetrics

def main():
    input_data_path = "original_data/"
    output_data_path = "distorted_data/"
    key_in = ""

    while key_in != "x":
        print("MENU")
        print("input path:  " + input_data_path)
        print("output path: " + output_data_path)
        print()
        print(" x - exit")
        print(" 1 - Distort Data")
        print(" 2 - Train Data")
        print(" 3 - Test Model")
        print(" 4 - HME Recognition")
        print(" 5 - change input path")
        print(" 6 - change output path")
        print("Input: ", end="")
        key_in = input()

        if key_in == "1":
            distortData(input_data_path, output_data_path)

        elif key_in == "2":
            trainModel(output_data_path, epochs = 10, output_file_name = "model.pt")
    
        elif key_in == "3":
            print('-- Choose which model to test --')
            print(" 1 - Negative log likelihood loss criterion function (with distorted data)")
            print(" 2 - Cross entropy loss criterion function (with distorted data)")
            print(" 3 - Negative log likelihood loss criterion function (no distorted data)")
            print(" x - exit")

            key_in = input()
            model = 'Trained_models/model_NLL.pt'
            isTest = True

            if key_in == '1':
                pass
            elif key_in == '2':
                model = 'Trained_models/model_cross-entropy.pt'
            elif key_in == '3':
                model = 'Trained_models/model_NLL_No-Distortion.pt'
            elif key_in == 'x':
                key_in = ''
                isTest = False
            else:
                print('Incorrect input. Using default NLL model')

            if isTest:
                TestMetrics(model)


        elif key_in == "4":
            PredictHME()

        elif key_in == "5":
            input_data_path = input()

        elif key_in == "6":
            output_data_path = input()

if __name__ == "__main__":
    main()
