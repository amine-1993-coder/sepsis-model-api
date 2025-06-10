from connexion.resolver import RestyResolver
import connexion
import os

app = connexion.App(__name__, specification_dir='swagger/')

if __name__ == '__main__':
  app.add_api('modelserving_app.yaml', resolver=RestyResolver('api'))
  app.run(port=9090)

