document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const conceptsContainer = document.getElementById('concepts-container');
    const conceptTemplate = document.getElementById('concept-template');
    const wordItemTemplate = document.getElementById('word-item-template');
    
    // Function to render concepts
    function renderConcepts(concepts) {
        // Clear existing concepts
        conceptsContainer.innerHTML = '';
        
        concepts.forEach(concept => {
            // Clone the concept template
            const conceptElement = conceptTemplate.content.cloneNode(true);
            
            // Set concept title and variance
            conceptElement.querySelector('.concept-title').textContent = `Concept ${concept.id}: ${concept.title}`;
            conceptElement.querySelector('.concept-variance').textContent = `${concept.varianceExplained.toFixed(1)}% variance`;
            
            // Container for words
            const conceptWordsContainer = conceptElement.querySelector('.concept-words');
            
            // Add words with their weights
            concept.words.forEach(wordData => {
                const wordElement = wordItemTemplate.content.cloneNode(true);
                wordElement.querySelector('.word-text').textContent = wordData.word;
                wordElement.querySelector('.weight-fill').style.width = `${wordData.weight * 100}%`;
                wordElement.querySelector('.word-weight-value').textContent = wordData.weight.toFixed(2);
                
                conceptWordsContainer.appendChild(wordElement);
            });
            
            // Create word cloud visualization
            createWordCloud(conceptElement.querySelector('.concept-visualization'), concept.words);
            
            // Set related events count
            conceptElement.querySelector('.event-count').textContent = concept.relatedEventsCount;
            
            // Add event listener to details button
            conceptElement.querySelector('.concept-details-button').addEventListener('click', () => {
                // In a real application, this would navigate to a details page or open a modal
                alert(`Viewing details for Concept ${concept.id}: ${concept.title}`);
            });
            
            // Add the concept to the container
            conceptsContainer.appendChild(conceptElement);
        });
    }
    
    // Create word cloud visualization
    function createWordCloud(container, words) {
        const cloudContainer = document.createElement('div');
        cloudContainer.style.width = '100%';
        cloudContainer.style.height = '100%';
        cloudContainer.style.display = 'flex';
        cloudContainer.style.justifyContent = 'center';
        cloudContainer.style.alignItems = 'center';
        cloudContainer.style.flexWrap = 'wrap';
        cloudContainer.style.padding = '10px';
        
        words.forEach(wordData => {
            const wordElement = document.createElement('span');
            wordElement.classList.add('word-cloud-word');
            wordElement.textContent = wordData.word;
            
            // Scale font size based on weight
            const fontSize = 14 + (wordData.weight * 24);
            wordElement.style.fontSize = `${fontSize}px`;
            
            // Use a consistent color palette with weight-based opacity
            const hue = (words.indexOf(wordData) * 30) % 360;
            const saturation = 70 + Math.floor(Math.random() * 15);
            const lightness = 45 + Math.floor(Math.random() * 10);
            wordElement.style.color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            
            // Add a small random offset for more natural placement
            wordElement.style.transform = `rotate(${Math.random() * 10 - 5}deg)`;
            
            // Add click event
            wordElement.addEventListener('click', () => {
                alert(`Selected word: "${wordData.word}" with weight ${wordData.weight.toFixed(2)}`);
            });
            
            cloudContainer.appendChild(wordElement);
        });
        
        container.appendChild(cloudContainer);
    }
    
    // Fetch concepts data from the server
    async function fetchConceptsData() {
        try {
            // Show loading state
            conceptsContainer.innerHTML = '<div class="loading-indicator">Loading SVD concepts data...</div>';
            
            const response = await fetch('/svd/query');
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update the summary statistics if they exist in the response
            if (data.summary) {
                const summaryCounts = document.querySelectorAll('.summary-value');
                if (summaryCounts.length >= 3) {
                    summaryCounts[0].textContent = data.summary.conceptsCount || 0;
                    summaryCounts[1].textContent = data.summary.varianceExplained ? 
                        `${data.summary.varianceExplained.toFixed(1)}%` : '0%';
                    summaryCounts[2].textContent = data.summary.documentsCount || 0;
                }
            }
            
            // Render the concepts
            renderConcepts(data.concepts || []);
            
        } catch (error) {
            console.error('Error fetching SVD concepts data:', error);
            conceptsContainer.innerHTML = `
                <div class="error-container">
                    <h3>Error Loading Data</h3>
                    <p>Unable to load SVD concepts. Please try again later.</p>
                    <p class="error-details">${error.message}</p>
                    <button id="retry-button">Retry</button>
                </div>
            `;
            
            // Add retry functionality
            document.getElementById('retry-button')?.addEventListener('click', fetchConceptsData);
        }
    }
    
    // Initialize the page by fetching data from the server
    fetchConceptsData();
});