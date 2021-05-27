import argparse


def avg_gating(path):
   
    avg = 0.
    count = 0.
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()[1:-1]
            line = line.split(',')
            for num in line:
                count += 1
                num = float(num)
                avg += num
    print("average gate value:", avg / count)


def norm(path):
    visual_avg = 0.
    count = 0.
    with open(path + 'norm_visual.txt') as f:
        for line in f.readlines():
            line = line.strip()[1:-1]
            line = line.split(',')
            for num in line:
                count += 1
                num = float(num)
                visual_avg += num
    visual_avg = visual_avg / count

    text_avg = 0.
    count = 0.
    with open(path + 'norm_text.txt') as f:
        for line in f.readlines():
            line = line.strip()[1:-1]
            line = line.split(',')
            for num in line:
                count += 1
                num = float(num)
                text_avg += num
    text_avg = text_avg / count

    print("norm average visual norm:", visual_avg)
    print("norm average text norm:", text_avg)
    print("percentage:", visual_avg/text_avg)


def outlier(path):
    significant = 0
    blind = 0
    aware = 0
    total = 0
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()[1:-1]
            line = line.split(',')
            for num in line:
                total += 1
                num = float(num)
                if num > 1e-5:
                    significant += 1
                if num > 1e-10:
                    aware += 1
                elif num == 0:
                    blind += 1
    print("In total gate value number:", total)
    print(">1e-10 rate and blind rate:", aware / total * 100, blind / total * 100)
    print(">1e-5 outlier rate:", significant / total)
    print("awareness:", (1 - (blind / total)) * 100)

if __name__ == '__main__':
    # visual_awareness()
    parser = argparse.ArgumentParser(description='')
    # fmt: off
    parser.add_argument('--input', required=True)
    # fmt: on
    args = parser.parse_args()

    avg_gating(args.input)
    # norm(args.input)
    # outlier(args.input)
