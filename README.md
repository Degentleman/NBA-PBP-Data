# NBA-PBP-Data

This is a project I started to analyze NBA play-by-play data using Python. My goal was to be able to parse an array of play-by-play data for a given matchup between two teams and build an adversarial synergy graph -- the approach was inspired by the paper above, titled "Weighted synergy graphs for effective team formation with
heterogeneous ad hoc agents." While I found the learning process challenging, I was able to employ the basic approach, ableit in a more "creative" fashion.

Creating an adversarial synergy graph requires lineup scores as training data. The points scored and allowed between two "adversarial" basketball teams are used to "guesses" the ideal graph structure that corresponds with the spread in the score (this approach assumes the teams are comprised of five people each). If there's a transitive property that explains the difference in score, manipulating a series of unweighted graphs can provide insight about agent-by-agent performance, leading to better insight about "optimal" team formation. While I'm new to Python, I've enjoyed the challenge of figuring out how to put the approach down into code, and I'm looking forward to updating this project throughout the 2018-2019 season.

Please read the research papers included in the master branch to familiarize yourself with the maths.

Any feedback is welcome!
