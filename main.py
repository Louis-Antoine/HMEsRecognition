from distortData import distortData
from TrainModel import trainModel

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
        print(" 3 - Test Data")
        print(" 4 - Load Model")
        print(" 5 - change input path")
        print(" 6 - change output path")
        print("Input: ", end="")
        key_in = input()

        if key_in == "1":
            distortData(input_data_path, output_data_path)

        elif key_in == "2":
            trainModel(output_data_path, epochs = 10, output_file_name = "model_crossentropy.pt")
    
        elif key_in == "3":
            pass

        elif key_in == "4":
            pass

        elif key_in == "5":
            input_data_path = input()

        elif key_in == "6":
            output_data_path = input()

if __name__ == "__main__":
    main()
