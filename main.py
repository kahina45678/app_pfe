__version__ = "1.0.3"


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from PIL import Image as PILImage
from PIL import ImageOps
from kivy.uix.relativelayout import RelativeLayout
import numpy as np
import cv2
import tensorflow as tf
import sqlite3
import re
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
import urllib.parse

con = sqlite3.connect("maladies.db")
cur = con.cursor()


# Chemin vers le modèle TensorFlow Lite
MODEL_PATH = 'model.tflite'

# Classes de maladies des plantes
CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
dico = {
	
    'Apple___Apple_scab': 'tavelure,pomme',
    'Apple___Black_rot': 'pourriture noire,pomme',
    'Apple___Cedar_apple_rust': 'rouille,pomme',
    'Apple___healthy': 'pommier en bonne santé',
    'Blueberry___healthy': 'myrtille en bonne santé',
    'Cherry_(including_sour)___Powdery_mildew': 'oïdium,cerise',
    'Cherry_(including_sour)___healthy': 'cerise en bonne santé',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'tache cercospora,maïs',
    "Corn_(maize)___Common_rust_": 'rouille commune,maïs',
    'Corn_(maize)___Northern_Leaf_Blight': 'tache foliaire septoria,maïs',
    'Corn_(maize)___healthy': 'maïs en bonne santé',
    'Grape___Black_rot': 'pourriture noire,vigne',
    'Grape___Esca_(Black_Measles)': 'esca,vigne',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'tache foliaire,vigne',
    'Grape___healthy': 'vigne en bonne santé',
    'Orange___Haunglongbing_(Citrus_greening)': 'huanglongbing,orange',
    'Peach___Bacterial_spot': 'tache bactérienne,pêche',
    'Peach___healthy': 'pêche en bonne santé',
    'Pepper,_bell___Bacterial_spot': 'tache bactérienne,poivron',
    'Pepper,_bell___healthy': 'poivron en bonne santé',
    'Potato___Early_blight': 'alternariose,pomme de terre',
    'Potato___Late_blight': 'mildiou,pomme de terre',
    'Potato___healthy': 'pomme de terre en bonne santé',
    'Raspberry___healthy': 'framboise en bonne santé',
    'Soybean___healthy': 'soja en bonne santé',
    'Squash___Powdery_mildew': 'oïdium,courge',
    'Strawberry___Leaf_scorch': 'brûlure des feuilles,fraise',
    'Strawberry___healthy': 'fraise en bonne santé',
    'Tomato___Bacterial_spot': 'tache bactérienne,tomate',
    'Tomato___Early_blight': 'alternariose,tomate',
    'Tomato___Late_blight': 'mildiou,tomate',
    'Tomato___Leaf_Mold': 'moisissure,tomate',
    'Tomato___Septoria_leaf_spot': 'tache septoria,tomate',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'acariens rouges,tomate',
    'Tomato___Target_Spot': 'tache cible,tomate',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'feuille jaune enroulée,tomate',
    'Tomato___Tomato_mosaic_virus': 'mosaïque,tomate',
    'Tomato___healthy': 'tomate en bonne santé',
}







class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.clearcolor = (0.7569, 0.8588, 0.702, 1)

        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.result_label = Label(text='[b]Identification des maladies des plantes[/b]', size_hint_y=None, height=50, font_size=32, halign='center', markup=True, font_name='rr.ttf')
        layout.add_widget(self.result_label)

        self.result_label = Label(text='[b][i][color=#D90368]Photographiez vos plantes pour identifier leurs maladies ![/color][/i][/b]', size_hint_y=None, height=30, font_size=25, halign='center', markup=True, font_name='ds.ttf')
        layout.add_widget(self.result_label)

        layout_image = BoxLayout(padding=(0, 0, 0, 30))
        self.image_view = Image(source='plante7.png', size=(700, 700))
        layout_image.add_widget(self.image_view)
        layout.add_widget(layout_image)

        self.prediction_label = Label(text='', size_hint_y=None, height=30, font_size=18, halign='center', markup=True)
        layout.add_widget(self.prediction_label)

        self.res_label = Label(text='', size_hint_y=None, height=30, font_size=18, halign='center', markup=True)
        layout.add_widget(self.res_label)  # Placé en dessous de prediction_label

        self.wikipedia_link_label = Label(text='', size_hint_y=None, height=30, font_size=18, halign='center', markup=True)
        layout.add_widget(self.wikipedia_link_label)

        button_layout = RelativeLayout(size_hint=(1, 0.2))

        buttons_box = BoxLayout(orientation='horizontal', spacing=20, size_hint=(None, None), size=(256, 128))
        buttons_box.pos_hint = {'center_x': 0.5, 'center_y': 1}

        capture_button = Button(background_normal='camera.png', background_down='camera.png', on_release=self.capture_photo, size_hint=(None, None), size=(256, 256))
        buttons_box.add_widget(capture_button)

        choose_button = Button(background_normal='upload.png', background_down='upload.png', on_release=self.choose_file, size_hint=(None, None), size=(128, 128))
        buttons_box.add_widget(choose_button)

        button_layout.add_widget(buttons_box)

        layout.add_widget(Widget())  # Espace vide pour centrer les boutons
        layout.add_widget(button_layout)

        self.add_widget(layout)

    def capture_photo(self, instance):
        self.manager.current = 'camera_screen'

    def choose_file(self, instance):
        self.manager.current = 'file_chooser_screen'

    def preprocess_image(self, image_path):
        if not isinstance(image_path, str):
            image_path = str(image_path)

        image = cv2.imread(image_path)

        # Définition des seuils pour la suppression des couleurs indésirables
        lower_blue = np.array([90, 0, 0])  
        upper_blue = np.array([255, 70, 70])  
        lower_pink = np.array([0, 0, 100])  
        upper_pink = np.array([70, 70, 255])  
        lower_violet = np.array([100, 0, 100])  
        upper_violet = np.array([255, 70, 255])  

        # Suppression des pixels de couleur bleue, rose et violet
        mask_blue = cv2.inRange(image, lower_blue, upper_blue)
        mask_pink = cv2.inRange(image, lower_pink, upper_pink)
        mask_violet = cv2.inRange(image, lower_violet, upper_violet)
        mask_combined = cv2.bitwise_or(mask_blue, mask_pink, mask_violet)
        image_filtered = cv2.bitwise_and(image, image, mask=255 - mask_combined)

        # Redimensionnement de l'image à la taille souhaitée
        resized_image = cv2.resize(image_filtered, (224, 224))

        # Normalisation de l'image
        thresholded = resized_image.astype(np.float32) / 255.0

        # Ajout d'une dimension pour correspondre à l'entrée du modèle
        thresholded = np.expand_dims(thresholded, axis=0)

        return thresholded


    def run_model(self, thresholded):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], thresholded)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        

        prediction_index = np.argmax(output)
        prediction_class = CLASSES[prediction_index]
        res = dico[prediction_class]

        confidence = output[0][prediction_index] * 100

        self.prediction_label.text = f'Prédiction : {res}\nConfiance : {confidence:.2f}%'

        maladie, plante = self.extraire_maladie_et_plante(res)
        self.bdd(maladie, plante)

        if maladie:
            wikipedia_link = self.get_wikipedia_url(maladie)
            link_text = f'Pour plus d\'informations, visitez le lien suivant : [ref={wikipedia_link}][color=0000FF]{wikipedia_link}[/color][/ref]'
            self.wikipedia_link_label.text = link_text
        else:
            self.wikipedia_link_label.text = ''

        return res

    def get_wikipedia_url(self, search_query):
        search_query_encoded = urllib.parse.quote(search_query)
        url = f"https://fr.wikipedia.org/w/index.php?search={search_query_encoded}"
        return url

    def extraire_maladie_et_plante(self, res):
        regex = r'^([^,]+),(.+)$'


        match = re.search(regex, res)
        if match:
            maladie = match.group(1)
            plante = match.group(2)
            
            print(plante, maladie)
            return maladie, plante
        else:
            return None, None

    def bdd(self, maladie, plante):
        query = "SELECT rem, dose FROM maladies WHERE nom = '{}' AND plante = '{}'".format(maladie, plante)
        cur.execute(query)
        results = cur.fetchall()

        if results:
            for result in results:
                rem, dose = result
                self.res_label.text += f'\n- Remède : {rem}, Dose : {dose}'
        else:
            self.res_label.text = ''

    def go_to_home(self, instance):
        self.manager.current = 'home_screen'



class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        return_button = Button(background_normal='retour.png', background_down='retour.png', on_release=self.go_to_home, size_hint=(None, None), size=(32, 32),pos_hint={'left': 1})

        layout.add_widget(return_button)

        self.camera = Camera(resolution=(640, 480), size_hint=(1, 0.7), play=True)
        layout.add_widget(self.camera)
        
        button_layout = RelativeLayout(size_hint=(1, 0.3))
        capture_button = Button(background_normal='obj.png', background_down='obj.png', on_release=self.capture_image, size_hint=(None, None), size=(128, 128))
        capture_button.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        button_layout.add_widget(capture_button)
        
        layout.add_widget(button_layout)
        self.add_widget(layout)

    def capture_image(self, instance):
        self.camera.export_to_png('captured_image.png')
        image_path = 'captured_image.png'
        self.manager.get_screen('home_screen').image_view.source = image_path
        image_data = self.manager.get_screen('home_screen').preprocess_image(image_path)
        self.manager.get_screen('home_screen').run_model(image_data)
        self.manager.current = 'home_screen'


    def go_to_home(self, instance):
        self.manager.current = 'home_screen'


class FileChooserScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        return_button = Button(background_normal='retour.png', background_down='retour.png', on_release=self.go_to_home, size_hint=(None, None), size=(32, 32), pos_hint={'left': 1})
        layout.add_widget(return_button)

        self.file_chooser = FileChooserListView()
        layout.add_widget(self.file_chooser)

        button_layout = BoxLayout(size_hint=(1, None), height=128, padding=(10, 10, 10, 0))
        choose_button = Button(background_normal='verifier.png', background_down='verifier.png', on_release=self.choose_file, size_hint=(None, None), size=(128, 128))
        button_layout.add_widget(Widget())
        button_layout.add_widget(choose_button)
        button_layout.add_widget(Widget())
        layout.add_widget(button_layout)

        self.error_label = Label(text='', size_hint_y=None, height=30, font_size=18, halign='center', markup=True)
        layout.add_widget(self.error_label)

        self.add_widget(layout)

    def choose_file(self, instance):
        selected_file = self.file_chooser.selection
        if selected_file:
            image_path = selected_file[0]
            self.manager.get_screen('home_screen').image_view.source = image_path
            image_data = self.manager.get_screen('home_screen').preprocess_image(image_path)
            self.manager.get_screen('home_screen').run_model(image_data)
            self.manager.current = 'home_screen'
        else:
            self.error_label.text = '[color=#FF0000]Erreur : Aucun fichier sélectionné ![/color]'

    def go_to_home(self, instance):
        self.manager.current = 'home_screen'


class PlantDiseaseApp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        self.home_screen = HomeScreen(name='home_screen')
        self.camera_screen = CameraScreen(name='camera_screen')
        self.file_chooser_screen = FileChooserScreen(name='file_chooser_screen')

        self.screen_manager.add_widget(self.home_screen)
        self.screen_manager.add_widget(self.camera_screen)
        self.screen_manager.add_widget(self.file_chooser_screen)

        self.screen_manager.current = 'home_screen'

        return self.screen_manager


if __name__ == '__main__':
    PlantDiseaseApp().run()
