import json
import argparse


def cli_main(path, output_name):
    in_file = read_file(path)
    names = {}
    with open('data/vatex.test.video') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            names[i] = line
    dc = {}
    for key, name in names.items():
        dc[name] = in_file[key]

    with open(output_name, 'w') as fp:
        json.dump(dc, fp)


def read_file(path):
    sys_toks = {}

    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')

            if line[0][:2] == 'H-':
                idx = int(line[0][2:])
                sys_toks[idx] = line[2]

    return sys_toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VMT')
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    output_name = args.dir + '/submission.json'
    cli_main(args.dir + '/results.out', output_name)

    # verify output
    # data = json.load(open('official_submission.json'))
    # my_data = json.load(open(output_name))

    # for d in data.keys():
    #     print(data[d], my_data[d])
