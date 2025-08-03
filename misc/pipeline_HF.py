from transformers import pipeline, Conversation
import gradio as gr

classifier = pipeline(task="sentiment-analysis")

output = classifier("""

I AM SO DUMB
                                             
""")

print(output)

summarizer = pipeline(task="summarization")

output = summarizer("""
New Starbucks CEO Brian Niccol won’t be a constant presence at its Seattle headquarters where he takes the helm next month. Instead, he’s going to commute weekly from his California home.

The setup was revealed last week in Niccol’s offer letter, which is giving him a “small remote office” at his home in Newport Beach, California, and not requiring him to permanently relocate to the coffee chain’s Seattle offices more than 1,000 miles away. Starbucks is giving him a corporate jet to use to commute back and forth.

“Brian Niccol has proven himself to be one of the most effective leaders in our industry, generating significant financial returns over many years,” a Starbucks spokesperson said in a comment to CNN. “We’re confident in his experience and ability to serve as the leader of our global business and brand, delivering long-term, enduring value for our partners, customers and shareholders.”

But Niccol’s private jet perk has brought some attention to the climate change implications of those flights and Starbucks’ projection as an environmentally friendly business, which recently rolled out new cups that use less plastic and eliminated plastic straws.

Jet travel, whether via large commercial jet or small private jet, is a major source of carbon emissions, responsible for about 800 million tons of carbon dioxide annually or more than 2% of total global energy-related emissions, according to the International Energy Agency.
                    
""")

print(output)