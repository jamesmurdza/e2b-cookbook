import fs from 'node:fs'

import { Anthropic } from '@anthropic-ai/sdk'
import { Sandbox } from '@e2b/code-interpreter'
import { OutputMessage } from '@e2b/code-interpreter'

import * as dotenv from 'dotenv'

dotenv.config()

//const MODEL_NAME = 'claude-3-opus-20240229'
const MODEL_NAME = 'claude-3-5-sonnet-20241022'

// Path inside the sandbox where the LLM is instructed to save the visualization.
const REMOTE_IMAGE_PATH = '/home/user/image.png'

const SYSTEM_PROMPT = `
## your job & context
you are a python data scientist. you are given tasks to complete and you run python code to solve them.
- the python code runs in jupyter notebook.
- every time you call \`execute_python\` tool, the python code is executed in a separate cell. it's okay to multiple calls to \`execute_python\`.
- create visualizations using matplotlib or any other visualization library and save them to a file at \`${REMOTE_IMAGE_PATH}\` (e.g. \`plt.savefig("${REMOTE_IMAGE_PATH}")\`). don't rely on inline display.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
- you can run any python code you want, everything is running in a secure sandbox environment.
`

const tools: Array<Anthropic.Tool> = [
    {
        name: 'execute_python',
        description: 'Execute python code in a Jupyter notebook cell and returns any result, stdout, stderr, display_data, and error.',
        input_schema: {
            type: 'object',
            properties: {
                code: {
                    type: 'string',
                    description: 'The python code to execute in a single cell.'
                }
            },
            required: ['code']
        }
    }
]


async function codeInterpret(codeInterpreter: Sandbox, code: string): Promise<string> {
    console.log('Running code interpreter...')

    const exec = await codeInterpreter.runCode(code, {
        onStderr: (msg: OutputMessage) => console.log('[Code Interpreter stderr]', msg),
        onStdout: (stdout: OutputMessage) => console.log('[Code Interpreter stdout]', stdout),
    })

    if (exec.error) {
        console.log('[Code Interpreter ERROR]', exec.error)
        throw new Error(exec.error.value)
    }
    return exec.logs.stdout.join('')
}



const client = new Anthropic()

async function processToolCall(codeInterpreter: Sandbox, toolName: string, toolInput: any): Promise<string> {
    if (toolName === 'execute_python') {
        return await codeInterpret(codeInterpreter, toolInput['code'])
    }
    return ''
}

async function chatWithClaude(codeInterpreter: Sandbox, userMessage: string): Promise<string> {
    console.log(`\n${'='.repeat(50)}\nUser Message: ${userMessage}\n${'='.repeat(50)}`)

    console.log('Waiting for Claude to respond...')
    const message = await client.messages.create({
        model: MODEL_NAME,
        system: SYSTEM_PROMPT,
        max_tokens: 4096,
        messages: [{ role: 'user', content: userMessage }],
        tools: tools,
    })

    console.log(`\nInitial Response:\nStop Reason: ${message.stop_reason}`)

    if (message.stop_reason === 'tool_use') {
        const toolUse = message.content.find(block => block.type === 'tool_use') as Anthropic.ToolUseBlock
        if (!toolUse){
            console.error('Tool use block not found in message content.')
            return ''
        }

        const toolName = toolUse.name
        const toolInput = toolUse.input

        console.log(`\nTool Used: ${toolName}\nTool Input: ${JSON.stringify(toolInput)}`)

        const toolResult = await processToolCall(codeInterpreter, toolName, toolInput)
        console.log(`Tool Result: ${toolResult}`)
        return toolResult
    }
    throw new Error('Tool use block not found in message content.')
}


async function run() {
    const codeInterpreter = await Sandbox.create()

    try {
        await chatWithClaude(
            codeInterpreter,
            'Calculate value of pi using monte carlo method. Use 1000 iterations. Visualize all point of all iterations on a single plot, a point inside the unit circle should be orange, other points should be grey.'
        )
        // The LLM was instructed to save the visualization to a file in the sandbox.
        // Download that file instead of reading rich results from the code interpreter.
        const png = await codeInterpreter.files.read(REMOTE_IMAGE_PATH, { format: 'bytes' })
        fs.writeFileSync('image.png', Buffer.from(png))
        console.log('Saved visualization to image.png')
    } catch (error) {
        console.error('An error occurred:', error)
        throw error;
    } finally {
        await codeInterpreter.kill()
    }
}

run()
