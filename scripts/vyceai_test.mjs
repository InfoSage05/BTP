import "dotenv/config";
import OpenAI from "openai";

if (!process.env.VYCEAI_API_KEY) {
  throw new Error("VYCEAI_API_KEY is missing from the project-root .env file.");
}

const client = new OpenAI({
  apiKey: process.env.VYCEAI_API_KEY,
  baseURL: "https://vyceai.com/v1",
});

const models = await client.models.list();
console.log("Available models:");
for (const model of models.data ?? []) console.log(`- ${model.id}`);

const response = await client.chat.completions.create({
  model: process.env.VYCEAI_MODEL || "auto",
  messages: [{ role: "user", content: "Say hello in one sentence." }],
});

console.log("\nResponse:");
console.log(response.choices[0]?.message?.content ?? "(empty response)");
