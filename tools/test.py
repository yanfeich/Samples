import os
ff = os.listdir('../src')
ff = [i for i in ff if os.path.isdir(os.path.join('../src', i))]

for f in ff:
    print(f'- [{f}](./src/{f}/README.md)')
