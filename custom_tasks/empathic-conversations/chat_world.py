#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.crowdsourcing.utils.worlds import CrowdOnboardWorld, CrowdTaskWorld  # type: ignore
from parlai.core.worlds import validate  # type: ignore
from parlai.agents._custom.remote import RemoteAgent

from joblib import Parallel, delayed  # type: ignore

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import project_credentials as pc  # project credentials for database and Remote Agent

from typing import List, Union

import json
import yaml

class MultiAgentDialogOnboardWorld(CrowdOnboardWorld):
    def __init__(self, opt, agent):
        super().__init__(opt, agent)
        self.opt = opt

    def parley(self):
        self.agent.agent_id = "Onboarding Agent"
        self.agent.observe({"id": "System", "text": "Welcome onboard!"})
        x = self.agent.act(timeout=self.opt["turn_timeout"])
        self.agent.observe(
            {
                "id": "System",
                "text": "Thank you for your input! Please wait while "
                "we match you with another worker...",
                "episode_done": True,
            }
        )
        self.episodeDone = True


class MultiAgentDialogWorld(CrowdTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, shared=None):
        # Add passed in agents directly.
        self.agents: List[Union["Agent", RemoteAgent]] = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("max_turns", 5)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.opt = opt

        # May want to remove PyMYSQL DBAPI for use on GCP.
        conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset:{encoding}'.format(
            user=pc.EC2_USER, password=pc.EC2_PASSWORD, host=pc.EC2_HOST, 
            port=pc.EC2_PORT, db=pc.EC2_SCHEMA, encoding = 'utf-8')
        
        # Empathic conversations 2 research database
        self.ec2_engine = create_engine(conn_string)

        # Local Mephisto db
        self.local_engine = create_engine(f'sqlite:///{pc.LOCAL_SQLITE_PATH}')

        # Make sure that the agents are in the agent table in the ec_2 schema, research database
        self.add_agents_db()
        # Add a conversation to the "conversation" table
        self.conversation_id = self.add_conversation_db()
        # Add two rows to conversation_agent table
        self.merge_conversation_agents_db(self.conversation_id)
        
        # A Mephisto thing...
        for idx, agent in enumerate(self.agents):
            agent.agent_id = f"Chat Agent {idx + 1}"

    def add_agents_db(self) -> None:
        for agent in self.agents:
            if isinstance(agent, RemoteAgent):
                id = self.agent_db_setup(agent.agent_name_for_db, type='c')

            else:
                # Get worker_id that was sent via query string. This is located in the Mephisto local db
                query = text('''
                    select worker_name
                    from workers as w
                    inner join agents as a on a.worker_id=w.worker_id
                    where a.unit_id = :unit_id;''')

                with self.local_engine.begin() as conn:
                    result = conn.execute(query, {'unit_id': agent.mephisto_agent.unit_id})
                    worker_name = result.fetchone()[0]

                print(f'Worker Name: {worker_name} Unit ID: {agent.mephisto_agent.unit_id}')

                agent.agent_name_for_db = worker_name
                id = self.agent_db_setup(agent.agent_name_for_db, type='w')
            
            # Set the ec2_agent_id to the primary key of the agent (in the ec2) db.
            agent.ec2_agent_id = id


    def agent_db_setup(self, name: str, type: Union['w', 'c'], remove_sandbox_postfix=True) -> int:
        if remove_sandbox_postfix and name[-8:] == '_sandbox':
            name = name[:-8]

        query = text('''
            select agent_id
            from agent
            where description = :name;''')

        with self.ec2_engine.begin() as conn:
            result = conn.execute(query, {'name': name}).fetchone()
        
        # If agent is NOT IN DB add it.
        if result is None:
            query = text('''
                insert into agent (agent_type, description) 
                values (:type, :name);''')
            
            with self.ec2_engine.begin() as conn:
                conn.execute(query, {'name': name, 'type': type})
                # We then get the primary key. It is not a composite key so -> [0]
                agent_id = conn.execute(text('select last_insert_id();')).fetchone()[0]

        else:
            agent_id = result[0]

        return agent_id


    def add_conversation_db(self) -> int:
        """
        This will add a new conversation to the table but will set article_id to NULL. 
        We do not know the exact article right now and will have to fill that out using the Qualtrics data.
        """
        # Read the config file (so that it can be put in db)
        with open(pc.YAML_MEPHISTO_CONFIG_FILE, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Convert the YAML data to JSON format
        json_data = json.dumps(yaml_data, indent=2)

        query = text('''insert into conversation (article_id, mephisto_settings) values (:article, :ms);''')
        
        with self.ec2_engine.begin() as conn:
            conn.execute(query, {'article': None, 'ms': json_data})
            conv_id = conn.execute(text('select last_insert_id();')).fetchone()[0]

        return conv_id

    def merge_conversation_agents_db(self, conversation_id: int) -> None:
        query = text('''
            insert into conversation_agent (conversation_id, agent_id, agent_unit_id)
            values (:cid, :aid, :auid);''')

        for agent in self.agents:
            with self.ec2_engine.begin() as conn:
                params = {
                    'cid': conversation_id,
                    'aid': agent.ec2_agent_id,
                    'auid': agent.mephisto_agent.unit_id if not isinstance(agent, RemoteAgent) else None
                }
                conn.execute(query, params)

    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        self.current_turns += 1
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=self.opt["turn_timeout"])
                if self.send_task_data:
                    acts[index].force_set(
                        "task_data",
                        {
                            "last_acting_agent": agent.agent_id,
                            "current_dialogue_turn": self.current_turns,
                            "utterance_count": self.current_turns + index,
                        },
                    )
                self.add_utterance_db(agent_id=agent.ec2_agent_id, utterance=acts[index].get('text'))
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent
                self.add_utterance_db(agent_id=agent.ec2_agent_id, utterance=acts[index].get('text'))
            if acts[index]["episode_done"]:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))

        if self.current_turns >= self.max_turns:
            self.episodeDone = True
            for agent in self.agents:
                if isinstance(agent, RemoteAgent):
                    agent.observe({"text": "[DONE]"})
                    _ = agent.act()
                    continue
                agent.observe(
                    {
                    "id": "Coordinator",
                    "text": 
                        f'''    
                        You are done with the conversation. 
                        Please enter the following code into the below 
                        input box and continue with the rest of the survey.
                        
                        CODE: {agent.mephisto_agent.unit_id}
                        '''
                    }
                )

    def add_utterance_db(self, agent_id: int, utterance: str) -> None:
        query = text('''
        insert into utterance(conversation_id, agent_id, content)
        values (:cid, :aid, :content)''')
        
        params = {'cid': self.conversation_id, 'aid': agent_id, 'content': utterance}
        
        with self.ec2_engine.begin() as conn:
            conn.execute(query, params)

        print(f'Added utterance to db: "{utterance}"')

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        return {"example_key": "example_value"}

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MultiAgentDialogOnboardWorld(opt, agent)


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    print(f"Validating onboarding data {data}")
    return True


def get_world_params():
    return {"agent_count": 2}


def make_world(opt, agents):
    bots = []

    while len(agents) + len(bots) < 2:
        bot = RemoteAgent({"host_bot": pc.HOST_BOT, 
                           "port_bot": pc.PORT_BOT})
        
        # agent_name_for_db should match the agent description in the database.
        bot.agent_name_for_db = pc.BOT_NAME

        # This is a hack to skip the OverWorld and TaskWorld by 
        # sending dummy messages. OverWorld accept any message
        # and TaskWorld accept only "begin" as the identifier.
        bot.observe({"text": "dummy"})
        _ = bot.act()
        bot.observe({"text": "begin"})
        _ = bot.act()

        bots.append(bot)

    # Combine agents with bots
    agents.extend(bots)

    return MultiAgentDialogWorld(opt, agents)
