Read AGENTS.md, then MCP_SESSION.md and TASK.md, and call `status`. Follow the AGENTS.md
workflow{% if agent_formal and skill_library %}: start from the nearest controller in the library, change it
for this task, and practise with it in every episode; when no episode remains, follow
`status.next_step` and end your turn{% elif agent_formal %}: solve the task by hand, automate it, and practise with your
controller in every episode; when no episode remains, follow `status.next_step` and end
your turn{% elif skill_library %}: start from the nearest controller in the library and change it for
this task{% endif %}.
