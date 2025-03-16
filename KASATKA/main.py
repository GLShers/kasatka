# main.py
from sound_classifier import SoundClassifier
from config import *

if __name__ == "__main__":
    try:
        classifier = SoundClassifier()
        
        if not classifier.load_model():
            print("Обучение новой модели...")
            classifier = SoundClassifier(
                data_dir='Learning',
                valid_dir='Valid',
                test_dir='Test'
            )
            classifier.train(num_epochs=48)
            classifier.save_model()
            classifier.test()
            
        classifier.start_recording()
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")