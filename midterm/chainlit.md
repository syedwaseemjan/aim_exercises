# LangChain powered RAG that can answer queries related to [Airbnb 10-k Filings from Q1, 2024](https://s26.q4cdn.com/656283129/files/doc_financials/2024/q1/fdb60f7d-e616-43dc-86ef-e33d3a9bdd05.pdf)

3 important queries this model has been specifically adjusted for are

Q1 "What is Airbnb's 'Description of Business'?"
Q2 "What was the total value of 'Cash and cash equivalents' as of December 31, 2023?"
Q3 "What is the 'maximum number of shares to be sold under the 10b5-1 Trading plan' by Brian Chesky?"

Total time: 6 hours
1. 15 minutes to configure HF space, setupt the repo, copy the boilerplate code and run the chainlit app locally. I copied the boilerplate code from Week 4 Day 1 exercise.
2. 15 minutes to run the endpoints I had configured previously. They were paused.
    1. Snowflake/snowflake-arctic-embed-m
    2. NousResearch/Meta-Llama-3-8B-Instruct
3. 30 minutes to fix the issue with pdf downloading. Was trying to trace what was causing the issue with PyMuPDFLoader but couldn't. Tried to download the pdf with requests lib and traced the issue to the missing User-Agent header.
4. 1 hour to fix the error `inputs tokens + max_new_tokens must be <= 1512. Given: 1584 inputs tokens and 512 max_new_tokens`.

    Things I tried

    1. Reduced the number of documents returned by Retriever K=2.
    2. Increased the `Max Number of Tokens (per Query)` to 4000. HF took 15 minutes to apply the changes.
    3. Increased the `Max Input Length (per Query)` to 2000. HF took 15 minutes to apply the changes.
    4. Reduce `chunk_size` to 500 from 1000 for RecursiveCharacterTextSplitter.

    2, 3 and 4 fixed the error, reverted the 1 as it was causing wrong results.

4. With the initial settings, specifically for RecursiveCharacterTextSplitter, I was able only to get the first question right. Answers to both 2 and 3 questions were wrong. Spent 4 hours in fixing the answers to all the questions.

     Things I tried

    1. Replaced the embedding model Snowflake/snowflake-arctic-embed-m with text-embedding-3-small. Answer to second model got fixed but 3rd and 1st got wrong.
    2. Reduced `chunk_size` to 200 from 500. 1st and 3rd answers were still wrong.
    3. Reduce `chunk_overlap` to 10 from 30. 1st and 3rd answers were still wrong.
    4. Reduce `chunk_overlap` to 0 from 30. 2nd and 3rd answers were right, only first was wrong.
    5. Added `length_function = tiktoken_len` RecursiveCharacterTextSplitter. All answers were right.

5. 30 minutes to fix `PermissionError: [Errno 13] Permission denied: './data/airbnb2020ipo.pdf'` when deploying on HF.