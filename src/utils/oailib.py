import asyncio
import json
from os import getenv

import diskcache as dc
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm


class AsyncOpenAIClient:
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv("OPENROUTER_API_KEY"),
        )
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()


class OpenAIClient:
    def __init__(self):
        self.client = None

    def __enter__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv("OPENROUTER_API_KEY"),
        )
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


async def simple_query_async(
    query,
    model="openai/gpt-4o-mini",
    system_msg=None,
    client=None,
    with_cache=True,
    retry=3,
    delay=1,
    fallback_models=["nousresearch/hermes-3-llama-3.1-70b"],
    **kwargs,
):
    if client is None:
        raise ValueError("Client must be provided")

    # Cache logic remains the same
    if with_cache:
        key = "#".join([str(x) for x in [query, model, system_msg]])
        cache = dc.Cache(".cache")
        if key in cache and isinstance(cache[key], str) and len(cache[key]) > 1:
            return cache[key]

    # Message assembly remains the same
    messages = [{"role": "system", "content": system_msg}] if system_msg else []
    messages.append({"role": "user", "content": query})

    models = [model] + fallback_models

    for model in models:
        out = None
        while retry > 0:
            try:
                completion = await client.chat.completions.create(
                    model=model, messages=messages, max_completion_tokens=8000, **kwargs
                )
                out = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {e}")
                out = None
                retry -= 1
                await asyncio.sleep(delay)
        if out is not None:
            break

    if with_cache and out is not None:
        cache[key] = out

    return out


async def _gather_queries(queries, model, system_msg, qps, with_cache, verbose, **kwargs):
    in_flight_requests = 0
    start_time = asyncio.get_event_loop().time()

    async def _query_with_tracking(query, index, client, progress_bar):
        nonlocal in_flight_requests, start_time

        # Calculate the ideal time this request should start
        ideal_start_time = start_time + (index / qps)

        # Wait until the ideal start time
        now = asyncio.get_event_loop().time()
        if ideal_start_time > now:
            await asyncio.sleep(ideal_start_time - now)

        in_flight_requests += 1
        progress_bar.set_postfix({"W": in_flight_requests})

        try:
            result = await simple_query_async(
                query, model, system_msg, client, with_cache, **kwargs
            )
            if result is None:
                result = "<No response>"
                if verbose:
                    tqdm.write(f"Query {index}: No response")
            elif verbose:
                tqdm.write(f"Query {index} result: {result[:50]}...")
            return result
        finally:
            in_flight_requests -= 1
            progress_bar.set_postfix({"W": in_flight_requests})

    async with AsyncOpenAIClient() as client:
        tasks = []
        with tqdm(total=len(queries)) as progress_bar:
            for i, query in enumerate(queries):
                task = asyncio.create_task(_query_with_tracking(query, i, client, progress_bar))
                task.add_done_callback(lambda p: progress_bar.update())
                tasks.append(task)
            results = await asyncio.gather(*tasks)
    return results


def grouped_generate(
    queries,
    model="openai/gpt-4o-mini",
    system_msg=None,
    qps=8,
    with_cache=True,
    verbose=False,
    **kwargs,
):
    key = json.dumps((queries, model, system_msg))
    cache = dc.Cache(".cache")
    if key in cache:
        print("All queries are in cache. Returning cached results.")
        return cache[key]

    outputs = asyncio.run(
        _gather_queries(queries, model, system_msg, qps, with_cache, verbose, **kwargs)
    )

    if with_cache:
        cache[key] = outputs

    return outputs


def single_generate(query, model="openai/gpt-4o-mini", system_msg=None, with_cache=True, **kwargs):
    async def _single_query():
        async with AsyncOpenAIClient() as client:
            return await simple_query_async(query, model, system_msg, client, with_cache, **kwargs)

    return asyncio.run(_single_query())


def stream(query, model="openai/gpt-4o-mini", system_msg=None, **kwargs):

    messages = [{"role": "system", "content": system_msg}] if system_msg else []
    messages.append({"role": "user", "content": query})

    with OpenAIClient() as client:
        try:
            streams = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=8000, stream=True, **kwargs
            )

            for chunk in streams:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")

        except Exception as e:
            print(f"Error: {e}")


def check_equal(answer1, answer2):
    # If both answers are lists, compare each pair of answers using grouped_query
    if isinstance(answer1, list) and isinstance(answer2, list):
        if len(answer1) != len(answer2):
            return False  # If the lengths are different, the answers are not equivalent

        # Generate the query for each pair
        queries = [
            f"====ANSWER1====\n{ans1}\n====ANSWER2====\n{ans2}\n====\nAre these two answers equivalent(has the same result)? Only answer 'yes' or 'no'. nothing else."
            for ans1, ans2 in zip(answer1, answer2)
        ]

        # Use grouped_query to process all queries in parallel
        results = grouped_generate(queries)

        # Check if all responses are "yes"
        return ["yes" in result.strip().lower() for result in results]

    # Fallback to single string comparison if both answers are not lists
    else:
        resp = single_generate(
            f"====ANSWER1====\n{answer1}\n====ANSWER2====\n{answer2}\n====\nAre these two answers equivalent(has the same result)? Only answer 'yes' or 'no'. nothing else."
        )
        resp = resp.strip().lower()
        if "yes" in resp:
            return True
        elif "no" in resp:
            return False
        else:
            raise ValueError(f"Invalid response from OpenAI API: {resp}")


if __name__ == "__main__":
    answer1 = """8) Counting the unique values of $n$, we get:
    4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 39, 49, 55, 65, 77

    Therefore, there are 18 possible values of $n$.

    <<ANSWER HERE>>18<<ANSWER HERE>>"""

    answer2 = """
    Therefore, I believe we can confidently say that there are 18 possible values of n.

    </contemplator>

    <final_answer>
    The number of possible values of n is <<18>>.
    </final_answer>"""

    print(check_equal(answer1, answer2))
    print(check_equal([answer1] * 10, [answer2] * 10))

    queries = [
        "What is AIss?",
    ] * 30
    results = grouped_generate(queries)
    for i, result in enumerate(results):
        print(f"Query {i+1}: {result}")
