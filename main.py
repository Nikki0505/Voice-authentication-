# Software starts with main.py


from interface import unauthorized
from model import speaker_identifier
#from model_testing import verifyUser

if __name__ == "__main__":

    #predict the user using speaker_identifier method from model
    pred = speaker_identifier()

    if pred == 1:
        print("You are authorized!")
        #if authorized start interacting and launch voice assistant
        exec(open(r"C:\Users\singh\Downloads\Voice-Biometrics-main\interface.py").read())
    else:
        print("Unauthorized !!")
        # exec(open(r"C:\Users\singh\Downloads\Voice-Biometrics-main\interface.py").read())
        #speak "You are unauthorized"
        unauthorized()
    
    











