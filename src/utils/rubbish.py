import yaml

with open('../config/vars.yml') as f:
    var = yaml.load(f, yaml.Loader)

print(var)