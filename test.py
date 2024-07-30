import os
ff = os.listdir('./')
ff = [i for i in ff if os.path.isdir(i) and not i.startswith('.')]

for f in ff:
    print(f'- [{f}](./{f}/README.md)')

print(ff)