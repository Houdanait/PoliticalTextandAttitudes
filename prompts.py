system_prompt = """You are a linguist analyzing dialogue transcripts for examples of "hedging" or "authority" pragmatic markers. The terms are provided in a list, and in the transcript the terms are capitalized within <> (e.g. <ABSOLUTELY>). Provide an answer for every single term in the list of "terms" you are given. Return your answer in the following JSON format:

-----BEGIN SAMPLE UNKNOWN NAME-----
"ANDERSON"
-----END SAMPLE UNKNOWN NAME-----

---BEGIN SAMPLE INPUT---
terms: ["believe", "maybe", "certainly", "best"]
transcript:
Speaker 1: "In the run for the presidency."
Speaker 2: "I knew about one incident. Understand the whole time that he ran for office, I knew that he had had one liaison. It still -- it still tore me up, I mean, personally tore me up. Did I think that one liaison would disqualify him to be the president? You know, we've had great presidents who I would hope one liaison would not have -- have stopped from serving us. That's what I believed. And I believed that until, golly, <MAYBE> long after it made any sense to but, <CERTAINLY> long after -- I mean, long after he was out of the race. And so sometimes I had to, you know, bite my tongue. I talked a lot about his policies, which I still <BELIEVE> were the <BEST> policies and set the standard for the other candidates on a lot of issues -- health care being one of them, but environment and poverty and corporate interference with government. And I really believed that that I could talk about those things and mean every word that I was saying, and have him as an advocate for those issues and meaning that as well."
---END SAMPLE INPUT---

---BEGIN SAMPLE RESPONSE---
    {{
        "believe": "hedge",
        "maybe": "hedge",
        "certainly": "authority",
        "best": "none"
    }}
---END SAMPLE RESPONSE---"""


speaker_prompt = """You are a transcript annotator, and your job is to clarify who is speaking in each utterance in a dialogue.

-----BEGIN SAMPLE TRANSCRIPT-----
<SPEAKER: ANDERSON>: You're watching CNN. This is CONNECT THE WORLD. I'm Becky Anderson. Welcome back, it's just half past seven here in the UAE. Earlier I spoke to Martin Griffiths who's the U.N. Special Counsel Envoy for Yemen to the Security Council and I got his take on how he sees the situation on the ground right now.
<SPEAKER: MARTIN GRIFFITHS, U.N. SPECIAL COUNSEL ENVOY FOR YEMEN>: I think what is going on is the war that we all need to spend more time trying to resolve. And today and yesterday and the day before have been really serious reminders of that.
<SPEAKER: ANDERSON>: There's a huge trust deficit between the Hadi government and the Houthis. What is your message at this point?
<SPEAKER: GRIFFITHS>: I think, first of all, that let's welcome what has been happening in terms of those redeployments in the last few days, step one.
<SPEAKER: ANDERSON>: Has the West simplified what is going on in Yemen, to Yemen's detriment?
<SPEAKER: GRIFFITHS>: I am sure that those who say that the Yemen war is a proxy war are simplifying the events in Yemen.
<SPEAKER: ANDERSON>: Yes. You heard what Martin Griffiths said, does that echo what you've heard on the ground, Sam?
<SPEAKER: SAM KILEY, CNN SENIOR INTERNATIONAL CORRESPONDENT>: Yes. It does, very strongly I have to say, Becky. I mean we've concentrated our work in the Houthi area and then Al Bayda.
<SPEAKER: ANDERSON>: Yes. The Yemen war, another issue that "IREPORT" puts the UAE's ministry of state and foreign affairs, Anwar Gargash, who I spoke to very late last night, I asked him how he saw the efforts to end the bloodshed there. Let's have a listen.
<SPEAKER: ANWAR GARGASH, UAE MINISTER OF STATE FOR FOREIGN AFFAIRS>: Right now, we have a very hopeful sign in Yemen, imperfect I have to admit, difficult I have to admit. But again we have a sign with the Stockholm agreement. We have for example now the pull out on Hodeidah.
<SPEAKER: ANDERSON>: There is increasing international pressure to stop the U.S. and Europe's arms sales to the coalition. Trump has vetoed the U.S. bill in the Senate to that effect. Are you worried that you will lose all international support for the campaign in Yemen? And how are you preparing for that?
<SPEAKER: GARGASH>: I think the campaign in Yemen is basically on the cusp of a different phase right now. And I think if you really look, the coalition in the last two years has been the biggest supporter for a political process.
<SPEAKER: ANDERSON>: This has been a long, bloody, grinding war for years. I genuinely believe he hopes a political solution is nigh. Is it?
<SPEAKER: KILEY>: I think it's urgent, certainly from the evidence we saw on the ground they are on the brink of a cholera epidemic. We visited a cholera clinic in Hadja, which is basically a fairly middle-class looking town up in the mountains. It's a beautiful scenic place. There were children there and adult grown men facing cholera.
<SPEAKER: ANDERSON>: A series of excellent reports as I suggested from Sam. I know you would applaud the team that you work with on ground in Yemen, those reports out next week starting I hope on Sunday, thank you, sir.
<SPEAKER: KILEY>: Thank you.
<SPEAKER: ANDERSON>: Always a pleasure to have you back. Live from Abu Dhabi, you're watching CONNECT THE WORLD. I'm Becky Anderson for you.
-----END SAMPLE TRANSCRIPT-----

For each speaker in the transcript, there is a name listed, but these are not consistent. Your job is to return a JSON with consistent details so that multiple statements can be matched to the same speaker. The JSON should be a dictionary with the speaker as initially provided as the key, and the value should be a dictionary with the following keys:

* name: The full name of the speaker in the format "LASTNAME, FIRSTNAME." If this isn't known, use "UNKNOWN" 
*occupation: The speaker's occupation

Your options for occupation are:

* Unknown
* News Media
* American Politician - Republican
* American Politician - Democrat
* Government Official
* Military
* Other

Do not make wild guesses about the speaker's occupation.

-----BEGIN SAMPLE OUTPUT-----

{
"ANDERSON": {
"name": "ANDERSON, BECKY",
"occupation": "News Media"
},
"MARTIN GRIFFITHS, U.N. SPECIAL COUNSEL ENVOY FOR YEMEN": {
"name": "GRIFFITHS, MARTIN",
"occupation": "Government Official"
},
"GRIFFITHS": {
"name": "GRIFFITHS, MARTIN",
"occupation": "Government Official"
},
"SAM KILEY, CNN SENIOR INTERNATIONAL CORRESPONDENT": {
"name": "KILEY, SAM",
"occupation": "News Media"
},
"ANWAR GARGASH, UAE MINISTER OF STATE FOR FOREIGN AFFAIRS": {
"name": "GARGASH, ANWAR",
"occupation": "Government Official"
},
"GARGASH": {
"name": "GARGASH, ANWAR",
"occupation": "Government Official"
},
"KILEY": {
"name": "KILEY, SAM",
"occupation": "News Media"
}
}
-----END SAMPLE OUTPUT-----
"""


speaker_prompt_alt = """You are a transcript annotator, and your job is to clarify who is speaking in each utterance in a dialogue. You are given a specific speaker, and your job is to output a JSON with details about them.

-----BEGIN SAMPLE TRANSCRIPT-----
<SPEAKER: ANDERSON>: You're watching CNN. This is CONNECT THE WORLD. I'm Becky Anderson. Welcome back, it's just half past seven here in the UAE. Earlier I spoke to Martin Griffiths who's the U.N. Special Counsel Envoy for Yemen to the Security Council and I got his take on how he sees the situation on the ground right now.
<SPEAKER: MARTIN GRIFFITHS, U.N. SPECIAL COUNSEL ENVOY FOR YEMEN>: I think what is going on is the war that we all need to spend more time trying to resolve. And today and yesterday and the day before have been really serious reminders of that.
<SPEAKER: ANDERSON>: There's a huge trust deficit between the Hadi government and the Houthis. What is your message at this point?
<SPEAKER: GRIFFITHS>: I think, first of all, that let's welcome what has been happening in terms of those redeployments in the last few days, step one.
<SPEAKER: ANDERSON>: Has the West simplified what is going on in Yemen, to Yemen's detriment?
<SPEAKER: GRIFFITHS>: I am sure that those who say that the Yemen war is a proxy war are simplifying the events in Yemen.
<SPEAKER: ANDERSON>: Yes. You heard what Martin Griffiths said, does that echo what you've heard on the ground, Sam?
<SPEAKER: SAM KILEY, CNN SENIOR INTERNATIONAL CORRESPONDENT>: Yes. It does, very strongly I have to say, Becky. I mean we've concentrated our work in the Houthi area and then Al Bayda.
<SPEAKER: ANDERSON>: Yes. The Yemen war, another issue that "IREPORT" puts the UAE's ministry of state and foreign affairs, Anwar Gargash, who I spoke to very late last night, I asked him how he saw the efforts to end the bloodshed there. Let's have a listen.
<SPEAKER: ANWAR GARGASH, UAE MINISTER OF STATE FOR FOREIGN AFFAIRS>: Right now, we have a very hopeful sign in Yemen, imperfect I have to admit, difficult I have to admit. But again we have a sign with the Stockholm agreement. We have for example now the pull out on Hodeidah.
<SPEAKER: ANDERSON>: There is increasing international pressure to stop the U.S. and Europe's arms sales to the coalition. Trump has vetoed the U.S. bill in the Senate to that effect. Are you worried that you will lose all international support for the campaign in Yemen? And how are you preparing for that?
<SPEAKER: GARGASH>: I think the campaign in Yemen is basically on the cusp of a different phase right now. And I think if you really look, the coalition in the last two years has been the biggest supporter for a political process.
<SPEAKER: ANDERSON>: This has been a long, bloody, grinding war for years. I genuinely believe he hopes a political solution is nigh. Is it?
<SPEAKER: KILEY>: I think it's urgent, certainly from the evidence we saw on the ground they are on the brink of a cholera epidemic. We visited a cholera clinic in Hadja, which is basically a fairly middle-class looking town up in the mountains. It's a beautiful scenic place. There were children there and adult grown men facing cholera.
<SPEAKER: ANDERSON>: A series of excellent reports as I suggested from Sam. I know you would applaud the team that you work with on ground in Yemen, those reports out next week starting I hope on Sunday, thank you, sir.
<SPEAKER: KILEY>: Thank you.
<SPEAKER: ANDERSON>: Always a pleasure to have you back. Live from Abu Dhabi, you're watching CONNECT THE WORLD. I'm Becky Anderson for you.
-----END SAMPLE TRANSCRIPT-----


You are given a single, specific name attributed to a speaker. For the given name (in the sample, "ANDERSON", your job is to return a JSON with consistent details so that multiple statements can be matched to the same speaker. The JSON should be a dictionary in the following format:

* key: This should be the exact name you are given as the input
* values: name, occupation
   * name: The full name of the speaker in the format "LASTNAME, FIRSTNAME." If this isn't known, use "UNKNOWN" 
   * occupation: The speaker's occupation

Your options for occupation are:
* Unknown
* News Media
* American Politician - Republican
* American Politician - Democrat
* Government Official
* Military
* Other
Do not make wild guesses about the speaker's occupation.

-----BEGIN SAMPLE OUTPUT-----

{
"ANDERSON": {
"name": "ANDERSON, BECKY",
"occupation": "News Media"
}
-----END SAMPLE OUTPUT-----
"""