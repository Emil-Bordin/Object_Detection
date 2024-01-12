# Object Detection

## Einleitung
Dieses Repository enthält mehrere Python-Skripte, die zusammen ein umfassendes Projekt für Datenverarbeitung, Modellinitialisierung, Vorhersage und Visualisierung bilden. Jedes Skript hat eine spezifische Rolle im Gesamtkontext des Projekts.

## Dateistruktur
- `Data_Loading.py`: Dieses Skript ist verantwortlich für das Laden der Daten. Es enthält Funktionen, um die benötigten Datensätze für das Projekt zu importieren und vorzubereiten.
- `Data_Prediction.py`: In diesem Skript werden Vorhersagen mit dem trainierten Modell durchgeführt. Es enthält die Logik für die Anwendung des Modells auf neue oder Testdaten.
- `Data_Visualization.py`: Dieses Skript kümmert sich um die Visualisierung der Daten und Ergebnisse. Es verwendet verschiedene Diagramme und Grafiken, um Einblicke in die Daten und die Leistung des Modells zu geben.
- `Dataset_Class.py`: Hier wird eine benutzerdefinierte Dataset-Klasse für das Projekt definiert. Diese Klasse hilft beim effizienten Laden und Verarbeiten der Daten in einem formatierten Format.
- `Main.py`: Das Hauptskript, das den gesamten Workflow orchestriert. Es verbindet alle anderen Skripte und führt das gesamte Projekt aus.
- `Model_Initialization.py`: Dieses Skript ist zuständig für die Initialisierung und Definition des neuronalen Netzwerkmodells. Es enthält die Architektur und die Parameter des Modells.

## Funktionsweise
Jedes Skript ist modular aufgebaut und führt eine spezifische Aufgabe im Rahmen des Projekts aus. Die Skripte interagieren miteinander, um den gesamten Prozess von der Datenvorbereitung über das Training des Modells bis hin zur Vorhersage und Visualisierung zu ermöglichen.

## Output
- **Datea_Loading**: Importiert und bereitet die Datensätze vor.
- **Model_Initialization**: Definiert das neuronale Netzwerkmodell.
- **Data_Visualization**: Stellt Ergebnisse und Daten grafisch dar.
- **Dataset_Class**: Organisiert die Daten in einer benutzerfreundlichen Klasse.
- **Main**: Führt alle Prozesse aus und koordiniert das Projekt.
- **Data_prediction**: Wendet das Modell auf neue Daten an und generiert Vorhersagen.
