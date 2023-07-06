# Utilisez une image de base contenant Python
FROM python:3.9-slim

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installez les dépendances requises
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le reste des fichiers de votre application dans le conteneur
COPY . .

# Exposez le port sur lequel Flask sera exécuté
EXPOSE 5000

# Démarrez l'application Flask
CMD ["spark-submit", "server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]





