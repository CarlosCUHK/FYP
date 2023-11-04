from typing import Any, Dict, List, Optional, Tuple
import re

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM


LEADER_PROMPT = """You are in charge of a team of agents responsible for making RESTful API calls to address user queries. Your team is composed of three roles: Planner, API Selector, and Caller, each with specific responsibilities. Your ultimate objective is to successfully address user queries. As the team leader, you oversee the entire working process of your team. During this process, you'll encounter two key situations:

1. Situation 1 (Agents seeking clarification):
Be prepared for the possibility that the agents under your supervision may encounter difficulties and require clarification or have questions for users. You act as the intermediary between users and the various agents for communication. Other agents are not permitted to directly communicate with users without your assistance. If agents have questions, they should preface their queries with "I need user's clarification." Your role is to assess the issues and explanations provided by the agents. If you agree with them, you must formulate a question to ask the user for additional information. When formulating a question for the user, you should begin your response with "Question:". Conversely, if you believe the problem is not the user's fault, you can challenge the agent's question and respond independently, beginning with "Refutation."

2. Situation 2 (End of one working cycle):
Typically, addressing user queries involves multiple iterative steps. At the end of each iteration, it is essential to clarify the overall objective for your team. At the conclusion of each cycle, you will encounter a marker labeled "End of Cycle" If the user's query has been effectively resolved after a cycle, you can conclude the process and present the query's outcome, starting with "Final Answer:" On the other hand, if the user's goal remains unmet, you should convey the user's expectations to the Planner for additional planning, initiating your response with "Objective to Planner:"

Example:
User: Give me the introduction to the movie "Batman vs Flash: Dawn"
Planner: Search for the movie "Batman vs Flash: Dawn".
API Selector: GET /search/movie to search for the movie "Batman vs Flash: Dawn".
Caller: Execution Result: I cannot finish executing the plan without knowing some other information. I need user's clarification.
Question: Please double check if the movie title is correct. If the query is correct, could you give me additional information for searching?
User: Sorry, the title of the movie should be "Batman v Superman: Dawn of Justice". 
Caller: The id of the movie "Batman v Superman: Dawn of Justice" is 209112
End of Cycle
Objective to Planner: Please give give the introduction o the movie "Batman v Superman: Dawn of Justice"
Planner: (continue to do the work)

Begin!

chat history is shown below:
{agent_scratchpad}

"""


class TeamLeader(Chain):
    llm: BaseLLM
    scenario: str
    leader_prompt: str
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, scenario: str, leader_prompt=LEADER_PROMPT) -> None:
        super().__init__(llm=llm, scenario=scenario, leader_prompt=leader_prompt)

    @property
    def _chain_type(self) -> str:
        return "Team Leader"

    @property
    def input_keys(self) -> List[str]:
        return ["history"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    
    
    def _construct_scratchpad(
        self, history
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, current_history in enumerate(history):
            scratchpad += current_history + "\n"
        return scratchpad

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        scratchpad = self._construct_scratchpad(inputs['history'])

        leader_prompt = PromptTemplate(
            template=self.leader_prompt,
            input_variables=["agent_scratchpad"]
        )
        leader_chain = LLMChain(llm=self.llm, prompt=leader_prompt)
        leader_chain_output = leader_chain.run(agent_scratchpad=scratchpad)
        

        return {"result": leader_chain_output}
