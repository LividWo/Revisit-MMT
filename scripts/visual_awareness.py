import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    # fmt: off
    parser.add_argument('--input', required=True)
    # fmt: on
    args = parser.parse_args()
    blind = 0
    aware = 0
    total = 0
    with open(args.input) as f:
        for line in f.readlines():
            line = line.strip()[1:-1]
            line = line.split(',')
            for num in line:
                total += 1
                num = float(num)
                if num > 1e-10:
                    aware += 1
                elif num == 0:
                    blind += 1
    print(aware/total, blind/total)


if __name__ == '__main__':
    main()
