import requests
import json
import pandas as pd
import os
import re
# dataset parse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('-s', '--scale-factor', type=int, default=10)
args = parser.parse_args()

# url = 'http://127.0.0.1:8080/completion'
url = f'http://{args.host}:{args.port}/completion'
header = {"Content-type": "application/json"}

File_Dir = "."
subjects = sorted( [ f.split("_test.csv")[0] for f in os.listdir(os.path.join(File_Dir, "test")) if "_test.csv" in f], reverse=False )

choices = ["A", "B", "C", "D"]
k = 1 # 5-shot

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    if k == 1:
        prompt = "The following is an example of multiple choice question on {}:\n\n".format(format_subject(subject=subject).strip())
    else:
        prompt = "The following are {} examples of multiple choice questions on {}:\n\n".format(k, format_subject(subject=subject).strip())
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)

    prompt += f"We will give one another question on {format_subject(subject=subject).strip()} to you. You can think carefully and step by step, but do not offer any explanation. "

    return prompt

if __name__ == "__main__":
    correct_cnt = 0
    wrong_cnt = 0
    for subject in subjects[:]:
        print(subject)
        #if 'high_school_biology' not in subject:
        #    continue

        data_df = pd.read_csv(os.path.join(File_Dir, "test", subject+"_test.csv"), header=None)
        dev_df = pd.read_csv(os.path.join(File_Dir, "dev", subject + "_dev.csv"), header=None)[:k]

        answers = choices[:data_df.shape[1]-2]

        save = dict()
        with open("./"+subject+"_test.json", 'a', encoding='utf-8') as fw:
            cnt = 0
            for i in range(args.scale_factor):
            # for i in range(data_df.shape[0]):
                # get prompt and make sure it fits
                print("===========================Qustion " + str(i) + "=======================================")
                prompt_end = format_example(data_df, i, include_answer=False)

                kshot = 1
                train_prompt = gen_prompt(dev_df, subject, kshot)
                prompt = train_prompt \
                    + f"Please answer the following question as the same format in the previous {'example' if kshot == 1 else 'examples'}:\n\n" \
                    + prompt_end
                #prompt = prompt_end.strip()

                #if 'college biology' not in prompt:
                #    continue

                print('Prompt:', repr(prompt) + '\n')

                #real_ans = data_df.iloc[i, 5]
                #print(f'real_ans: {real_ans}')
                # with open(f'prompt/{subject}_{real_ans}.txt', 'w') as f:
                #     f.write(f'<Q> {prompt}')

                payload = {"prompt": prompt, "n_predict": 4, "temperature": 0, "repeat_penalty": 1.0}
                # print(payload)
                res = requests.post(url, json=payload, headers=header)
                json_res = json.loads(res.text)

                print('Response:', repr(json_res['content']) + '\n')

                re_res = re.search('[A-D]',json_res['content'].strip())
                if re_res:
                    save[str(cnt)] = re_res.group()
                else:
                    save[str(cnt)] = 'X'

                llm_ans = save[str(cnt)]
                real_ans = data_df.iloc[i, 5]
                if llm_ans == real_ans:
                    correct_cnt += 1
                else:
                    wrong_cnt += 1
                print(llm_ans, real_ans, correct_cnt, wrong_cnt, f'{correct_cnt / (correct_cnt + wrong_cnt):.3f}')
                print( "LLM answer " + save[str(cnt)] + "/[ " +  data_df.iloc[i, 5] + "] for ["  + str(cnt+1) +  "/" + str(data_df.shape[0]) \
                      + "] in [" + subject +"]" )
                cnt = cnt + 1

            json.dump(save, fw, indent=4, ensure_ascii=False)
            fw.write('\n')
        fw.close()

    total_cnt = wrong_cnt + correct_cnt
    print(f'Final result: #total={total_cnt}, #correct={correct_cnt}, #wrong={wrong_cnt}, score={correct_cnt * 100 / total_cnt:.1f}%')
