import shutil
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import re
import json
import os
import ffmpeg

from functools import reduce

# import tiktoken


openai_api_key = os.environ['OPEN_API_KEY']
model = 'gpt-3.5-turbo'


def get_timestamps(chunks_text):

    # Prompt to get title and summary for each chunk
    prompt_template = """Given the following subtitle text I want you to help get me the important parts and the timestamp of each
  and also make sure your output is in JSON format. The max amount of clips is 5 and the difference between the start_time and end_time 
  should be greater than 4 seconds so you can merge other subtitle section to create a longer clip. 
  
  and please just the json, don't append any additional text to your response:
  
  {text}

  Please return your answer in the following format:
  [
  {{
    "start_time": "00:00:00,000",
    "end_time": "00:00:05,060",
    "text": "The speaker recently bought a 3D printer."
  }},
  {{
    "start_time": "00:00:10,920",
    "end_time": "00:01:12,719",
    "text": "3D printers allow you to find pre-modeled designs online and print them easily."
  }},
  {{
    "start_time": "00:01:34,000",
    "end_time": "00:02:14,580",
    "text": "The speaker finds satisfaction in creating perfect fits for their belongings using "
  }}
]
  """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    # Define the LLMs
    llm = ChatOpenAI(temperature=0, model_name=model,
                     openai_api_key=openai_api_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    llm_chain_input = [{'text': t} for t in chunks_text]
    # Run the input through the LLM chain (works in parallel)
    resp = llm_chain.apply(llm_chain_input)
    return resp


def get_text_chunks(text):

    # tokenizer = tiktoken.encoding_for_model(model)

    # token_length = len(tokenizer.encode(text))

    # subtitle_len = int(re.findall("(^\d+)\n", text, re.MULTILINE)[-1])

    text = text.splitlines()

    endlines = [i for i in range(len(text)) if text[i] == ""]

    # text_length = "\n".join(text)

    _range = len(endlines) // 4

    cutoffs = [endlines[i] for i in range(0, len(endlines), _range)]

    chunks = []
    for endpoint in cutoffs:
        chunk = "\n".join(text[endpoint: endpoint+_range])
        chunks.append(chunk)

    cleaned_chunk = [re.sub("^\n", "", re.sub(
        "^\d+\n", "", c, flags=re.MULTILINE), flags=re.MULTILINE) for c in chunks]

    return cleaned_chunk


def trim(in_file, out_file, start, end):
    if os.path.exists(out_file):
        os.remove(out_file)

    input_stream = ffmpeg.input(in_file)

    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start, end=end).setpts(pts)
    audio = input_stream.filter_(
        "atrim", start=start, end=end).filter_("asetpts", pts)
    video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
    output = ffmpeg.output(video_and_audio, out_file, format="mp4")
    output.run()


def concatenate_clips(clips, output_path, _dir):
    merged_video_list_name = f"video_list.txt"

    with open(merged_video_list_name, 'w') as f:
        for video in clips:
            print(f"file {video}", file=f)
            # print(f"file {video}")

    ffmpeg.input(merged_video_list_name,
                 format='concat', safe=0).output(output_path, c='copy').run()
    

def timestamp_to_seconds(timestamp):
    h, m, s_m = timestamp.split(":")
    h = int(h) * 3600
    m = int(m) * 60
    s, ms = s_m.split(",")
    s = int(s)
    ms = int(ms) / 1000

    return h + m + s + ms


def get_clips(file, cuts):
    try:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        temp_clips = []

        in_file_probe_result = ffmpeg.probe(file)
        in_file_duration = in_file_probe_result.get(
            "format", {}).get("duration", None)
        # print(in_file_duration, timestamp_to_seconds(cuts[-1]['end_time']), cuts[-1]['end_time'])
        filtered_cuts = [cuts[0]]
        for cut in cuts[1: ]:
            if filtered_cuts[-1]["end_time"] == cut["start_time"]:
                filtered_cuts[-1]["end_time"] = cut["end_time"]
                continue
            filtered_cuts.append(cut)


        for i, clip in enumerate(filtered_cuts):
            start_time = timestamp_to_seconds(clip["start_time"])
            end_time = timestamp_to_seconds(clip["end_time"])

            print("the clip starts from", start_time, "and ends at", end_time, "for", clip["start_time"], clip["end_time"] , i)

            if int(end_time - start_time) < 1:
                continue

            clip_filename = os.path.join(temp_dir, f"tempclip_{i}.mp4")

            trim(file, clip_filename, start_time, end_time)
            temp_clips.append(clip_filename)

        concatenate_clips(temp_clips, "output.mp4", "temp")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=False, onerror=None)


if __name__ == "__main__":
    # The example here is using a relative path to movie.mp4
    # trim("test1.mp4", "out.mp4", 10, 18)
    data = open("test2.srt", 'r').read()

    print("infering clips . . . . .")
    chunks = get_text_chunks(data)
    print(chunks[-1])
    output = [json.loads(obj['text']) for obj in get_timestamps(chunks)]

    # with open("output.json", "r") as f:
    #     output = json.loads(f.read())

    with open("output.json", "w") as f:
        json.dump(output, f)

    clips = reduce(lambda x, y: x+y, output)

    print("extracting clips . . . . .")
    get_clips("test2.mp4", clips)
