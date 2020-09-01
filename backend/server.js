const express = require('express')
const path = require('path')
const morgan = require('morgan')
const cookieParser = require('cookie-parser')
const bodyParser = require('body-parser')
const formidable = require('formidable')
const fs = require('fs')
const cors = require('cors')
require('dotenv').config()

//app
const app = express()


// middlewares
app.use(morgan('dev'))
app.use(bodyParser.json())
app.use(cookieParser())
app.use(express.static(path.join(__dirname, 'base')));

//cors
app.use(cors())


//routes
app.get('/api/base', (req, res) => {
    res.json({
        "base": "ON BASE!"
    })
});

app.post('/api/draw', (req, res) => {
    const form = new formidable.IncomingForm();
    form.parse(req, function(err, fields, files) {
        var oldPath = files.draw.path;
        var rawData = fs.readFileSync(oldPath)
        if(rawData) {
            console.log('IMAGE');
            res.json({
                'success': rawData
            })
        }
    })
});

//port
const port = process.env.PORT || 8000
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
})