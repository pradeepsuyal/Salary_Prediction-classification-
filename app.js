Skip to content
Why GitHub? 
Team
Enterprise
Explore 
Marketplace
Pricing 
Search
Sign in
Sign up
dinkwiz
/
tableau_embed
Public
05
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
tableau_embed/app.js /
@dinkwiz
dinkwiz Add files via upload
Latest commit 90112f1 on 17 May 2020
 History
 1 contributor
49 lines (38 sloc)  1.33 KB
   
console.log('Is this working?');

let viz;

//Add Share Link to Tableau Public in here
const url = "https://public.tableau.com/views/Squirrels_15746293266160/Dashboard1?:display_count=y&:origin=viz_share_link";

const vizContainer = document.getElementById('vizContainer');
const options = {
    hideTabs: true,
    height: 1000,
    width: 1200,
    onFirstInteraction: function() {
        workbook = viz.getWorkbook();
        activeSheet = workbook.getActiveSheet();
        console.log("My dashboard is interactive");
    }
};

//create a function to generate the viz element
function initViz() {
    console.log('Executing the initViz function!');
    viz = new tableau.Viz(vizContainer, url, options);
}

// run the initViz function when the page loads
document.addEventListener("DOMContentLoaded", initViz);

const exportPDF = document.getElementById('exportPDF');
const exportImage = document.getElementById('exportImage');


//click on the pdf button to generate pdf of dashboard
function generatePDF() {
    viz.showExportPDFDialog()
}

exportPDF.addEventListener("click", function () {
    generatePDF();
  });

//click on image to generate image of dashboard
function generateImage() {
    viz.showExportImageDialog()
}

exportImage.addEventListener("click", function () {
    generateImage();
  });
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
