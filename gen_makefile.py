import os
ff = os.listdir('./src')

for ii in ff:
    jj = os.path.join('./src', ii)
    if not os.path.isdir(jj):
        continue

    cpp_files = os.listdir(jj)
    cpp_files = [i.split('.cpp')[0] for i in cpp_files if i.endswith('.cpp')]
    if len(cpp_files) == 0:
        continue

    makefile = os.path.join(jj, 'Makefile')
    lines = []
    for k in ['make', 'dev']:
        lines.append('{}:'.format(k))
        for cpp_file in cpp_files:
            lines.append('\tg++ -std=gnu++11 -I/usr/include/habanalabs -Wall -g -o {} {}.cpp -L/usr/lib/habanalabs/ -lSynapse -ldl'.format(cpp_file, cpp_file))
            lines.append('\tmkdir -p ../../bin')
            lines.append('\tmv {} ../../bin'.format(cpp_file))
    
    lines.append('clean:')
    for cpp_file in cpp_files:
        lines.append('\trm -rf ../../bin/{}'.format(cpp_file))
    lines = [i+'\n' for i in lines]

    with open(makefile, 'w') as fp:
        fp.writelines(lines)

