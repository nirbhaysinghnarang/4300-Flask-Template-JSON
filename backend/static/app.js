var elements = null;


///This will load stuff.
document.addEventListener('DOMContentLoaded', function() {
    elements = {
        searchInput: document.getElementById('search-input'),
        searchButton: document.getElementById('search-button'),
        resultsContainer: document.getElementById('results-container'),
        resultsHeading: document.getElementById('results-heading'),
        minYear: document.getElementById('min-year'),
        maxYear: document.getElementById('max-year'),
        minYearError: document.getElementById('min-year-error'),
        maxYearError: document.getElementById('max-year-error'),
        eventTemplate: document.getElementById('event-template'),
        redditPostTemplate: document.getElementById('reddit-post-template'),
        detailItemTemplate: document.getElementById('detail-item-template'), 
        socialMediaWeight: document.getElementById('weigh-social-media'),
        tfIdfMethod: document.getElementById('tfidf-method'),
        gloveMethod: document.getElementById('glove-method'),
        svdMethod: document.getElementById('svd-method')
    };

    console.log(elements)
    const configElement = document.getElementById('app-config');
    config = {
        mapboxToken: configElement ? configElement.getAttribute('data-mapbox-token') : '',
        apiEndpoint: '/historical-sites',
        mapStyle: 'mapbox://styles/nirbhaysinghnarang/clibkhjth02vf01quas1vbm0s',
        defaultZoom: 4
    };
    console.log(config)
    setupEventListeners()
});


function setupEventListeners() {
    elements.searchButton.addEventListener('click', performSearch);
    elements.searchInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            performSearch();
        }
    });
    elements.minYear.addEventListener('input', () => validateYearInput(elements.minYear, elements.minYearError));
    elements.maxYear.addEventListener('input', () => validateYearInput(elements.maxYear, elements.maxYearError));
}

function validateYearInput(inputElement, errorElement) {
    if (!validateYear(inputElement.value)) {
        inputElement.classList.add('error');
        errorElement.classList.add('visible');
        return false;
    } else {
        inputElement.classList.remove('error');
        errorElement.classList.remove('visible');
        return true;
    }
}

function validateYear(year) {
    if (year === "") {
        return true;
    }
    const yearRegex = /^\d{1,4}BC$|\d{1,4}$|\d{1,4}AD$/i;
    return yearRegex.test(year);
}

function performSearch() {
    const query = elements.searchInput.value.trim();

    const minYearValid = validateYearInput(elements.minYear, elements.minYearError);
    const maxYearValid = validateYearInput(elements.maxYear, elements.maxYearError);
    
    if (!minYearValid || !maxYearValid) {
        return; 
    }

    if (query === '') {
        showNoResults('Please enter a search term to discover historical events.');
        return;
    }

    showLoading();    
    const searchParams = new URLSearchParams({ query });
    if (elements.minYear.value) searchParams.append('minYear', elements.minYear.value);
    if (elements.maxYear.value) searchParams.append('maxYear', elements.maxYear.value);
    if(elements.socialMediaWeight.checked) searchParams.append('useReddit', 'true');


    if(elements.tfIdfMethod.checked) {
        embeddingMethod = "TF"
    } else if (elements.svdMethod.checked) {
        embeddingMethod = "SVD"
    } else if (elements.gloveMethod.checked) {
        embeddingMethod = "GLOVE"
    }

    searchParams.append('embeddingMethod', embeddingMethod);


    console.log(elements.tfIdfMethod.checked)
    console.log("FETCHING")
    fetch(`${config.apiEndpoint}?${searchParams.toString()}`)
        .then(response => {

     
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => displayResults(data))
        .catch(error => {
            console.error("Error fetching data:", error);
            showNoResults('An error occurred while searching. Please try again.');
        });
}

function showLoading() {
    elements.resultsContainer.innerHTML = '<div class="loading">Searching through history...</div>';
    elements.resultsHeading.style.display = 'block';
}

function showNoResults(message) {
    elements.resultsContainer.innerHTML = `<div class="no-results"><p>${message}</p></div>`;
    elements.resultsHeading.style.display = 'none';
}

function displayResults(data) {
    //scroll to bottom of document.getElementById(results-container)
    elements.resultsContainer.innerHTML = '';
    
    if (data.length === 0) {
        showNoResults('No historical events found matching your search.');
        return;
    }
    
    data.forEach((event, index) => {
        const resultElement = createResultElement(event, index);
        elements.resultsContainer.appendChild(resultElement);
        
        if (event.row && event.row.longitude && event.row.latitude) {
            setTimeout(() => {
                initializeMap(event, resultElement.querySelector('.map-container'), index);
                document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });

            }, 100);
        }
    });
}

function createResultElement(event, index) {
    const template = elements.eventTemplate.content.cloneNode(true);
    const resultItem = template.querySelector('.result-item');    
    resultItem.querySelector('.result-title').textContent = getEventTitle(event);
    resultItem.querySelector('.result-description').textContent = getEventDescription(event);    
    const metadataContainer = resultItem.querySelector('.result-metadata');
    populateMetadata(metadataContainer, event);    
    const significanceVal = calculateSignificance(event);
    const significancePct = (significanceVal / 10) * 100;
    resultItem.querySelector('.significance-fill').style.width = `${significancePct}%`;
    resultItem.querySelector('.significance-value').textContent = `${significanceVal.toFixed(1)}/10`;
    
    const expandButton = resultItem.querySelector('.expand-button');
    expandButton.addEventListener('click', () => toggleExpanded(expandButton));   
    
    //add theemz
    populateThemes(resultItem, event);
    populateSimilarEvents(resultItem, event);


    setupTabs(resultItem, event, index);    
    populateDetailTab(resultItem, event);
    populateTextTab(resultItem, 'context', getEventContext(event));
    populateTextTab(resultItem, 'consequences', getEventConsequences(event));
    populateTextTab(resultItem, 'significance', getEventSignificance(event));
    populateTextTab(resultItem, 'facts', getEventFacts(event));
    populateRedditTab(resultItem, event, index);
    
    return resultItem;
}
function populateThemes(resultItem, event) {
    const themesContainer = resultItem.querySelector('.themes-tags');
    if (!themesContainer) return;
    
    themesContainer.innerHTML = '';
    
    if (event.themes && event.themes.length > 0) {
        event.themes.forEach(theme => {
            const themeTag = document.createElement('span');
            themeTag.className = 'theme-tag';
            
            // Handle SVD themes which might have a special format
            if (typeof theme === 'string' && theme.startsWith('concept ')) {
                // For SVD, themes might be in format "Concept X: term1, term2, term3"
                themeTag.className += ' svd-theme';
                
                const parts = theme.split(':');
                if (parts.length > 1) {
                    const conceptName = parts[0].trim();
                    const terms = parts[1].trim();
                    
                    const conceptSpan = document.createElement('span');
                    conceptSpan.className = 'concept-name';
                    conceptSpan.textContent = conceptName;
                    
                    const termsSpan = document.createElement('span');
                    termsSpan.className = 'concept-terms';
                    termsSpan.textContent = terms;
                    
                    themeTag.appendChild(conceptSpan);
                    themeTag.appendChild(termsSpan);
                } else {
                    themeTag.textContent = theme;
                }
            } else {
                // For TF-IDF and GloVe, themes are simple terms
                themeTag.textContent = theme;
            }
            
            themesContainer.appendChild(themeTag);
        });
    } else {
        const noThemesSpan = document.createElement('span');
        noThemesSpan.className = 'no-themes';
        noThemesSpan.textContent = 'No themes available';
        themesContainer.appendChild(noThemesSpan);
    }
}

function populateSimilarEvents(resultItem, event) {
    const similarEventsContainer = resultItem.querySelector('.similar-events-list');
    if (!similarEventsContainer) return;
    
    similarEventsContainer.innerHTML = '';
    
    if (event.similar_documents && event.similar_documents.length > 0) {
        event.similar_documents.forEach(doc => {
            const similarEvent = document.createElement('div');
            similarEvent.className = 'similar-event';
            similarEvent.textContent = doc.document;
            similarEventsContainer.appendChild(similarEvent);
        });
    } else {
        const noSimilarEvents = document.createElement('div');
        noSimilarEvents.className = 'no-similar-events';
        noSimilarEvents.textContent = 'No similar events found';
        similarEventsContainer.appendChild(noSimilarEvents);
    }
}

function getEventTitle(event) {
    return event.row ? event.row['Name of Incident'] : 
        (event.document ? event.document.split('(')[0].trim() : 'Unknown Event');
}

function getEventDescription(event) {
    return event.row && event.row.description || 'No description available.';
}

function getEventContext(event) {
    return event.row && event.row.context;
}

function getEventConsequences(event) {
    return event.row && event.row.immediate_consequences;
}

function getEventSignificance(event) {
    return event.row && event.row.long_term_significance;
}

function getEventFacts(event) {
    return event.row && event.row.interesting_facts;
}

function getEventType(event) {
    return event.row && event.row['Type of Event'] || 'Unknown';
}

function getEventPerson(event) {
    return event.row && event.row['Important Person/Group Responsible'] || 'Not specified';
}

function getEventPopulation(event) {
    return event.row && event.row['Affected Population'] || 'Not specified';
}

function getEventImpact(event) {
    return event.row && event.row['Impact'] || 'Not specified';
}

function getEventOutcome(event) {
    return event.row && event.row['Outcome'] || 'Unknown';
}

function formatYear(event) {
    if (!event.row || !event.row.Year) return 'Unknown year';
    
    try {
        const yearNum = parseInt(event.row.Year);
        if (isNaN(yearNum)) return event.row.Year.toString();
        return yearNum > 0 ? `${yearNum} CE` : `${Math.abs(yearNum)} BCE`;
    } catch (e) {
        return event.row.Year.toString();
    }
}

function formatLocation(event) {
    if (!event.row) return 'Location unknown';
    
    const hasPlace = event.row['Place Name'] && 
                    event.row['Place Name'] !== 'undefined' && 
                    event.row['Place Name'] !== 'null';
    
    const hasCountry = event.row.Country && 
                       event.row.Country !== 'undefined' && 
                       event.row.Country !== 'null';
    
    if (hasPlace && hasCountry) {
        return `${event.row['Place Name']}, ${event.row.Country}`;
    } else if (hasPlace) {
        return event.row['Place Name'];
    } else if (hasCountry) {
        return event.row.Country;
    } else {
        return 'Location unknown';
    }
}

function calculateSignificance(event) {
    return event.score ? Math.round((event.score * 100)) / 10 : 0;
}

function formatRedditDate(timestamp) {
    if (!timestamp) return 'Unknown date';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
}

// UI interaction functions
function toggleExpanded(button) {
    const expandedDiv = button.nextElementSibling;
    const isHidden = expandedDiv.style.display === 'none' || expandedDiv.style.display === '';

    expandedDiv.style.display = isHidden ? 'block' : 'none';
    button.textContent = isHidden ? 'Hide Details ↑' : 'View Full Details ↓';
}

function initializeMap(event, mapContainer, index) {
    if (!mapContainer) return;
    
    try {
        const coords = [
            parseFloat(event.row.longitude),
            parseFloat(event.row.latitude)
        ];
        
        if (!coords[0] || !coords[1]) {
            mapContainer.style.display = 'none';
            return;
        }
        
        mapboxgl.accessToken = config.mapboxToken;
        const map = new mapboxgl.Map({
            container: mapContainer,
            style: config.mapStyle,
            center: coords,
            zoom: config.defaultZoom
        });
        
        new mapboxgl.Marker({ color: '#7d6852' })
            .setLngLat(coords)
            .addTo(map);
    } catch (error) {
        console.error('Error initializing map:', error);
        mapContainer.style.display = 'none';
    }
}

function createRedditPostElement(post) {
    const template = elements.redditPostTemplate.content.cloneNode(true);
    const postElement = template.querySelector('.reddit-post');
    postElement.querySelector('.reddit-post-title').textContent = `"${post.title}"`;
    postElement.querySelector('.reddit-post-author').textContent = `Posted by u/${post.author}`;
    postElement.querySelector('.reddit-post-date').textContent = formatRedditDate(post.created_utc);
    postElement.querySelector('.reddit-post-subreddit').textContent = post.subreddit;
    postElement.querySelector('.reddit-post-score').textContent = `⬆️ ${post.score}`;
    postElement.querySelector('.reddit-post-comments').textContent = `💬 ${post.num_comments}`;
    const linkElement = postElement.querySelector('.reddit-post-link');
    linkElement.href = post.url;
    linkElement.textContent = 'View on Reddit';
    return postElement;
}



function populateMetadata(container, event) {
    const yearDisplay = formatYear(event);
    appendMetaItem(container, '🗓️', yearDisplay);    
    const era = event.row && event.row.era || 'Unknown era';
    appendMetaItem(container, '🏛️', era);    
    const location = formatLocation(event);
    if (location) {
        appendMetaItem(container, '📍', location);
    }
}

function appendMetaItem(container, icon, text) {
    const metaItem = document.createElement('div');
    metaItem.className = 'meta-item';
    metaItem.innerHTML = `<i>${icon}</i> ${text}`;
    container.appendChild(metaItem);
}

function setupTabs(resultItem, event, index) {
    const tabs = resultItem.querySelectorAll('.tab');
    const tabContents = resultItem.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        const tabType = tab.getAttribute('data-tab');
        const contentId = `${tabType}-${index}`;
        const contentElement = resultItem.querySelector(`#${tabType}-content`);
        if (contentElement) {
            contentElement.id = contentId;
        }
        
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');            
            tabContents.forEach(content => content.classList.remove('active'));
            const selectedContent = resultItem.querySelector(`#${contentId}`);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }
        });
    });
}

function populateDetailTab(resultItem, event) {
    const detailItems = resultItem.querySelector('.detail-items');
    if (!detailItems) return;
    
    detailItems.innerHTML = '';
    
    addDetailItem(detailItems, 'Name of Incident', getEventTitle(event));
    addDetailItem(detailItems, 'Type of Event', getEventType(event));
    addDetailItem(detailItems, 'Date', formatYear(event));
    addDetailItem(detailItems, 'Location', formatLocation(event));
    addDetailItem(detailItems, 'Important Person/Group', getEventPerson(event));
    addDetailItem(detailItems, 'Affected Population', getEventPopulation(event));
    addDetailItem(detailItems, 'Impact', getEventImpact(event));
    addDetailItem(detailItems, 'Outcome', getEventOutcome(event));
}

function addDetailItem(container, label, value) {
    const template = elements.detailItemTemplate.content.cloneNode(true);
    const detailItem = template.querySelector('.detail-item');
    
    detailItem.querySelector('strong').textContent = label;
    detailItem.querySelector('span').textContent = value || 'Not specified';
    
    container.appendChild(detailItem);
}

function populateTextTab(resultItem, tabType, text) {
    const contentElement = resultItem.querySelector(`.${tabType}-text`);
    if (contentElement) {
        contentElement.textContent = text || `No ${tabType} information available for this event.`;
    }
}

function populateRedditTab(resultItem, event, index) {
    const redditContainer = resultItem.querySelector('.reddit-posts-container');
    if (!redditContainer) return;
    
    const redditPosts = event.row && event.row.reddit_posts;
    
    if (!redditPosts || redditPosts.length === 0) {
        redditContainer.innerHTML = '<div class="no-reddit-posts">No Reddit discussions found for this historical event.</div>';
        return;
    }
    
    redditContainer.innerHTML = '';
    
    redditPosts.forEach(post => {
        const postElement = createRedditPostElement(post);
        redditContainer.appendChild(postElement);
    });
}