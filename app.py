from connexion.resolver import RestyResolver
import connexion
import os

# Initialize Connexion app
app = connexion.App(__name__, specification_dir='swagger')
app.add_api('modelserving_app.yaml', resolver=RestyResolver('api'))

if __name__ == '__main__':
    app.run(port=9090)
